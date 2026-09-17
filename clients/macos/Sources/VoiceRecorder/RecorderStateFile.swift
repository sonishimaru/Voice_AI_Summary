import Foundation

/// JSON snapshot written to `recorder_state.json` (the "is it actually
/// recording right now" file) and appended, one line at a time, to
/// `recorder_events.jsonl` (the history of every state transition). This
/// is the only way anything outside the app -- the Python ingestion
/// worker, a health-check script, a person debugging a gap in the
/// transcript -- can tell whether recording is happening without asking
/// the menu bar. Field names/casing are spelled out via `CodingKeys`
/// (snake_case), the same pattern `SegmentWriter`'s `Sidecar` uses.
struct RecorderStateSnapshot: Encodable {
    var schema: Int = 1
    var state: String
    var since: String
    var resumeAt: String?
    var reason: String
    var pid: Int32
    var appVersion: String
    var updatedAt: String
    /// Only set on the `delete_recent` event; nil (omitted) otherwise.
    var minutes: Int?
    /// Only set on the `delete_recent` event; nil (omitted) otherwise.
    var files: Int?

    enum CodingKeys: String, CodingKey {
        case schema
        case state
        case since
        case resumeAt = "resume_at"
        case reason
        case pid
        case appVersion = "app_version"
        case updatedAt = "updated_at"
        case minutes
        case files
    }

    /// Hand-written so `resume_at` is always present (`null` when nil):
    /// the synthesized encoder would omit a nil Optional entirely, and the
    /// Python reader (`recorder_state.read_state`) treats the key as part
    /// of the schema. `minutes`/`files` stay omit-when-nil, as documented.
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(schema, forKey: .schema)
        try c.encode(state, forKey: .state)
        try c.encode(since, forKey: .since)
        try c.encode(resumeAt, forKey: .resumeAt)
        try c.encode(reason, forKey: .reason)
        try c.encode(pid, forKey: .pid)
        try c.encode(appVersion, forKey: .appVersion)
        try c.encode(updatedAt, forKey: .updatedAt)
        try c.encodeIfPresent(minutes, forKey: .minutes)
        try c.encodeIfPresent(files, forKey: .files)
    }
}

/// Owns `<stateDir>/recorder_state.json` and `<stateDir>/recorder_events.jsonl`.
///
/// Both files live in `Settings.stateDirectoryURL` (by default, the inbox
/// directory's parent). `recorder_state.json` is overwritten in place with
/// the latest snapshot; `recorder_events.jsonl` is append-only, one JSON
/// object per line, so a reader can replay the full history of pauses,
/// resumes, sleeps and deletes.
///
/// All file I/O happens on a private serial queue -- like `SegmentWriter`,
/// so writes never race each other -- except `writeSync`, which blocks
/// the caller; that's deliberate, for the process-termination path where
/// an `async`-dispatched write might never get to run.
final class RecorderStateFile {
    static let shared = RecorderStateFile()

    private let queue = DispatchQueue(label: "com.voiceaisummary.recorder.statefile")

    private init() {}

    static let appVersion: String = {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.1.0"
    }()

    /// Shared UTC `YYYY-MM-DDTHH:MM:SSZ` formatter. `SegmentWriter`'s
    /// sidecar `started_at_utc` field uses this exact same formatter
    /// (factored out here) so every timestamp the app writes to disk is
    /// formatted identically.
    static let iso8601UTCFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss'Z'"
        formatter.timeZone = TimeZone(identifier: "UTC")
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return formatter
    }()

    /// Overwrites `recorder_state.json` with the latest snapshot and
    /// appends the same snapshot to `recorder_events.jsonl`. Fire-and-forget
    /// from the caller's point of view; safe to call from the main actor.
    func write(_ snapshot: RecorderStateSnapshot) {
        queue.async {
            Self.writeStateLocked(snapshot)
            Self.appendEventLocked(snapshot)
        }
    }

    /// Synchronous variant for `RecordingController.prepareForQuit()`,
    /// where the process may be torn down before a `queue.async` block
    /// gets a chance to run.
    func writeSync(_ snapshot: RecorderStateSnapshot) {
        queue.sync {
            Self.writeStateLocked(snapshot)
            Self.appendEventLocked(snapshot)
        }
    }

    /// Overwrites `recorder_state.json` only, without appending to the
    /// event log -- used for the 60 s heartbeat while recording, which
    /// refreshes `updated_at` but is not a state transition and would
    /// otherwise add ~1,440 lines a day to a log that readers replay in
    /// full on every call.
    func writeStateOnly(_ snapshot: RecorderStateSnapshot) {
        queue.async {
            Self.writeStateLocked(snapshot)
        }
    }

    /// Appends to `recorder_events.jsonl` only, without touching the
    /// "current state" snapshot file -- used for `delete_recent`, which
    /// is an event but not a state transition.
    func appendEventOnly(_ snapshot: RecorderStateSnapshot) {
        queue.async {
            Self.appendEventLocked(snapshot)
        }
    }

    // MARK: - Locked helpers (only ever run on `queue`)

    private static func stateDirectoryURL() -> URL {
        let url = Settings.shared.stateDirectoryURL
        // Best-effort: in the default configuration this directory is the
        // inbox's parent, which already exists by the time recording has
        // ever started; a custom `stateDirPath` override might not.
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private static func writeStateLocked(_ snapshot: RecorderStateSnapshot) {
        let url = stateDirectoryURL().appendingPathComponent("recorder_state.json")
        do {
            let data = try JSONEncoder().encode(snapshot)
            try data.write(to: url, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        } catch {
            Log.controller.error("Failed to write recorder_state.json: \(error.localizedDescription, privacy: .public)")
        }
    }

    private static func appendEventLocked(_ snapshot: RecorderStateSnapshot) {
        let url = stateDirectoryURL().appendingPathComponent("recorder_events.jsonl")
        do {
            var line = try JSONEncoder().encode(snapshot)
            line.append(0x0A) // "\n"

            let fm = FileManager.default
            if !fm.fileExists(atPath: url.path) {
                if !fm.createFile(atPath: url.path, contents: nil, attributes: [.posixPermissions: 0o600]) {
                    Log.controller.error("Failed to create recorder_events.jsonl.")
                    return
                }
            }
            let handle = try FileHandle(forWritingTo: url)
            handle.seekToEndOfFile()
            handle.write(line)
            handle.closeFile()
        } catch {
            Log.controller.error("Failed to append to recorder_events.jsonl: \(error.localizedDescription, privacy: .public)")
        }
    }
}
