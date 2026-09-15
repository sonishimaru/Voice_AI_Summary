import Foundation
import AVFoundation

/// Identifies which of the two always-on tracks a writer is for. The raw
/// value is used verbatim as the `{source}` component of output filenames,
/// per spec (`mac_mic` / `mac_system`).
enum AudioSource: String {
    case mic = "mac_mic"
    case system = "mac_system"
}

/// Owns exactly one rolling AAC file for one track (mic or system audio):
/// opening, writing, rotating on a timer, and the "delete current segment"
/// privacy control.
///
/// All file I/O happens on a private serial queue so writes from the audio
/// callback path never race with rotation/close, and so the callback path
/// itself never blocks on disk I/O directly -- it just enqueues.
final class SegmentWriter {
    let source: AudioSource
    private let deviceID: String
    private let directory: URL
    private let sampleRate: Double
    private let channels: UInt32

    private let queue: DispatchQueue

    private var currentFile: AVAudioFile?
    private var currentTempURL: URL?
    private var currentFinalURL: URL?
    private var currentStartedAt: Date?

    init(source: AudioSource, deviceID: String, directory: URL, sampleRate: Double = 16_000, channels: UInt32 = 1) {
        self.source = source
        self.deviceID = deviceID
        self.directory = directory
        self.sampleRate = sampleRate
        self.channels = channels
        self.queue = DispatchQueue(label: "com.voiceaisummary.recorder.writer.\(source.rawValue)")
    }

    /// Opens a brand-new segment file. Call once after construction, and
    /// again after every `rotate()` / `deleteCurrentAndRestart()`.
    func open() {
        queue.async { [weak self] in
            self?.openLocked()
        }
    }

    /// Writes one already-converted PCM buffer (expected: 16 kHz mono
    /// Float32 non-interleaved, matching `sampleRate`/`channels`) to the
    /// currently open file. Safe to call from any thread; the actual disk
    /// write always happens on this writer's own serial queue.
    func write(_ buffer: AVAudioPCMBuffer) {
        queue.async { [weak self] in
            guard let self, let file = self.currentFile else { return }
            do {
                try file.write(from: buffer)
            } catch {
                Log.writer.error("Write failed for \(self.source.rawValue, privacy: .public): \(error.localizedDescription, privacy: .public)")
            }
        }
    }

    /// Closes the current file, finalizes it (rename off `.part` + writes
    /// the JSON sidecar), and immediately opens a fresh one. Called by the
    /// rotation timer every `rotationMinutes`.
    func rotate() {
        queue.async { [weak self] in
            self?.closeAndFinalizeLocked()
            self?.openLocked()
        }
    }

    /// Privacy control ("Delete last 15 minutes"): discards whatever has
    /// been recorded into the current, still-open segment instead of
    /// finalizing it, then immediately starts a new segment. Because files
    /// roll every `rotationMinutes`, this deletes at most that much audio.
    func deleteCurrentAndRestart() {
        queue.async { [weak self] in
            self?.discardLocked()
            self?.openLocked()
        }
    }

    /// Closes and finalizes the current file without opening a new one.
    /// Call when stopping recording entirely.
    func close() {
        queue.async { [weak self] in
            self?.closeAndFinalizeLocked()
        }
    }

    // MARK: - Locked helpers (only ever run on `queue`)

    private func openLocked() {
        let startedAt = Date()
        let timestamp = Self.filenameTimestampFormatter.string(from: startedAt)
        let stem = "\(source.rawValue)_\(deviceID)_\(timestamp)"
        let finalURL = directory.appendingPathComponent("\(stem).m4a")
        // Written under a `.part` name first; only renamed to the final
        // name once fully closed, so the Python worker never picks up a
        // half-written file (per spec).
        let tempURL = directory.appendingPathComponent("\(stem).m4a.part")

        // AAC-LC, ~32 kbps, mono, 16 kHz per spec. `AVAudioFile` encodes
        // compressed output from the PCM buffers passed to `write(from:)`
        // as long as `commonFormat`/`interleaved` below describe those
        // *input* buffers (not the encoded file itself).
        //
        // NOTE: verify this on a real Mac. If `AVAudioFile` refuses to
        // open an AAC destination this way (some AVFoundation versions are
        // picky about compressed formats vs. sample-rate/channel
        // combinations), fall back to Linear PCM settings and a `.caf`
        // extension instead -- nothing else in this class needs to change,
        // since callers only see `AudioSource`/directory, not the codec.
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: Int(channels),
            AVEncoderBitRateKey: 32_000
        ]

        do {
            let file = try AVAudioFile(forWriting: tempURL, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
            currentFile = file
            currentTempURL = tempURL
            currentFinalURL = finalURL
            currentStartedAt = startedAt
        } catch {
            Log.writer.error("Failed to open segment for \(self.source.rawValue, privacy: .public): \(error.localizedDescription, privacy: .public)")
            currentFile = nil
            currentTempURL = nil
            currentFinalURL = nil
            currentStartedAt = nil
        }
    }

    private func closeAndFinalizeLocked() {
        guard let tempURL = currentTempURL, let finalURL = currentFinalURL, let startedAt = currentStartedAt else {
            currentFile = nil
            return
        }
        // AVAudioFile has no explicit close(); dropping the last strong
        // reference flushes and closes the underlying file handle.
        currentFile = nil

        do {
            try FileManager.default.moveItem(at: tempURL, to: finalURL)
        } catch {
            Log.writer.error("Failed to finalize segment for \(self.source.rawValue, privacy: .public): \(error.localizedDescription, privacy: .public)")
            currentTempURL = nil
            currentFinalURL = nil
            currentStartedAt = nil
            return
        }

        writeSidecar(for: finalURL, startedAt: startedAt)

        currentTempURL = nil
        currentFinalURL = nil
        currentStartedAt = nil
    }

    private func discardLocked() {
        currentFile = nil
        if let tempURL = currentTempURL {
            try? FileManager.default.removeItem(at: tempURL)
        }
        currentTempURL = nil
        currentFinalURL = nil
        currentStartedAt = nil
    }

    private func writeSidecar(for audioURL: URL, startedAt: Date) {
        let sidecarURL = audioURL.deletingPathExtension().appendingPathExtension("json")
        let sidecar = Sidecar(
            source: source.rawValue,
            deviceID: deviceID,
            startedAtUTC: Self.iso8601UTCFormatter.string(from: startedAt),
            tzOffset: Self.currentTimeZoneOffsetString(),
            sampleRate: Int(sampleRate),
            channels: Int(channels),
            codec: "aac",
            appVersion: Self.appVersion
        )
        do {
            let data = try JSONEncoder().encode(sidecar)
            try data.write(to: sidecarURL, options: .atomic)
        } catch {
            Log.writer.error("Failed to write sidecar for \(self.source.rawValue, privacy: .public): \(error.localizedDescription, privacy: .public)")
        }
    }

    // MARK: - Formatting helpers

    /// `{YYYYMMDDTHHMMSSZ}` filename component, UTC.
    private static let filenameTimestampFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
        formatter.timeZone = TimeZone(identifier: "UTC")
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return formatter
    }()

    /// `started_at_utc` sidecar field, e.g. "2026-09-15T01:02:03Z".
    private static let iso8601UTCFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss'Z'"
        formatter.timeZone = TimeZone(identifier: "UTC")
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return formatter
    }()

    /// `tz_offset` sidecar field, e.g. "+09:00".
    private static func currentTimeZoneOffsetString() -> String {
        let seconds = TimeZone.current.secondsFromGMT()
        let sign = seconds >= 0 ? "+" : "-"
        let absSeconds = abs(seconds)
        let hours = absSeconds / 3600
        let minutes = (absSeconds % 3600) / 60
        return String(format: "%@%02d:%02d", sign, hours, minutes)
    }

    private static let appVersion: String = {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.1.0"
    }()
}

/// JSON sidecar written next to each finalized segment. Field names/casing
/// are spelled out via `CodingKeys` (snake_case) to match the spec exactly,
/// rather than relying on a global key-encoding strategy.
private struct Sidecar: Encodable {
    let source: String
    let deviceID: String
    let startedAtUTC: String
    let tzOffset: String
    let sampleRate: Int
    let channels: Int
    let codec: String
    let appVersion: String

    enum CodingKeys: String, CodingKey {
        case source
        case deviceID = "device_id"
        case startedAtUTC = "started_at_utc"
        case tzOffset = "tz_offset"
        case sampleRate = "sample_rate"
        case channels
        case codec
        case appVersion = "app_version"
    }
}
