import Foundation
import Combine
import AVFoundation
import AudioToolbox
import AppKit

/// Top-level state machine for the app: owns both capture pipelines and
/// their writers, the rotation timer, pause/resume, the "delete last 15
/// minutes" privacy control, and recovery from device/sleep/wake changes.
///
/// `@MainActor` because SwiftUI reads `@Published` properties on the main
/// thread; the two capture classes deliver buffers on their own background
/// queues and never touch `@Published` state directly -- only writers do
/// (and writers have their own queues), so there's no cross-actor hazard
/// there. Every Core Audio / NSWorkspace notification handled here hops
/// back onto the main queue before touching controller state.
@MainActor
final class RecordingController: ObservableObject {

    // `Equatable` is required (not auto-synthesized without the explicit
    // conformance) because `start()`/`pause()`/`resume()`/the recovery
    // paths, and the SwiftUI menu content, all compare `state` with `==`.
    enum State: Equatable {
        case recording
        case paused
        case stopped
    }

    @Published private(set) var state: State = .stopped
    @Published private(set) var recordingStartedAt: Date?

    private let settings = Settings.shared

    private var micCapture: MicCapture?
    private var systemCapture: ProcessTapCapture?

    // Writers are created once (on first `start()`) and live for the
    // whole recording session, including across pause/resume -- only the
    // capture pipelines above are torn down and rebuilt for pause/resume
    // and for device-change recovery. This avoids ever abandoning an
    // open (un-renamed) `.part` file.
    private var micWriter: SegmentWriter?
    private var systemWriter: SegmentWriter?

    private var rotationTimer: Timer?
    private var restartWorkItem: DispatchWorkItem?
    private var wasRecordingBeforeSleep = false

    // Core Audio default-device change listeners, kept as properties only
    // so they aren't deallocated; this controller lives for the process's
    // entire lifetime (owned by the App struct), so explicit teardown in
    // `deinit` is not required in practice.
    private var defaultInputListenerBlock: AudioObjectPropertyListenerBlock?
    private var defaultOutputListenerBlock: AudioObjectPropertyListenerBlock?
    private var sleepObserver: NSObjectProtocol?
    private var wakeObserver: NSObjectProtocol?
    private var configChangeObserver: NSObjectProtocol?

    init() {
        registerSystemObservers()
    }

    // MARK: - Public controls

    /// Starts (or resumes) recording. Safe to call repeatedly / from
    /// `resume()` and the various recovery paths below: if writers already
    /// exist from an earlier `start()` in this session, they are reused
    /// as-is (not recreated), so only the capture pipelines are (re)built.
    func start() {
        guard state != .recording else { return }

        do {
            try settings.ensureInboxDirectoryExists()
        } catch {
            Log.controller.error("Could not create inbox directory: \(error.localizedDescription, privacy: .public)")
            scheduleRetry()
            return
        }

        if micWriter == nil || systemWriter == nil {
            let deviceID = Self.sanitizedDeviceID()
            let mic = SegmentWriter(source: .mic, deviceID: deviceID, directory: settings.inboxURL)
            let system = SegmentWriter(source: .system, deviceID: deviceID, directory: settings.inboxURL)
            mic.open()
            system.open()
            micWriter = mic
            systemWriter = system
        }

        guard let mic = MicCapture(), let system = ProcessTapCapture() else {
            Log.controller.error("Failed to allocate capture objects.")
            scheduleRetry()
            return
        }
        mic.onBuffer = { [weak self] buffer in
            self?.micWriter?.write(buffer)
        }
        system.onBuffer = { [weak self] buffer in
            self?.systemWriter?.write(buffer)
        }

        do {
            try mic.start()
            try system.start()
        } catch {
            Log.controller.error("Failed to start capture: \(error.localizedDescription, privacy: .public)")
            scheduleRetry()
            return
        }

        micCapture = mic
        systemCapture = system
        state = .recording
        if recordingStartedAt == nil {
            recordingStartedAt = Date()
        }
        startRotationTimer()
        Log.controller.info("Recording started.")
    }

    /// Fully stops recording: tears down both capture pipelines AND
    /// finalizes both writers' current segments. Use `pause()` instead for
    /// a temporary stop that keeps the current segment open.
    func stop() {
        rotationTimer?.invalidate()
        rotationTimer = nil
        micCapture?.stop()
        systemCapture?.stop()
        micCapture = nil
        systemCapture = nil
        micWriter?.close()
        systemWriter?.close()
        micWriter = nil
        systemWriter = nil
        state = .stopped
        recordingStartedAt = nil
        Log.controller.info("Recording stopped.")
    }

    func pause() {
        guard state == .recording else { return }
        micCapture?.stop()
        systemCapture?.stop()
        micCapture = nil
        systemCapture = nil
        rotationTimer?.invalidate()
        rotationTimer = nil
        state = .paused
        Log.controller.info("Recording paused.")
    }

    func resume() {
        guard state == .paused else { return }
        start()
        Log.controller.info("Recording resumed.")
    }

    /// "Delete last 15 minutes" -- the minimal privacy control. Deletes
    /// the currently-open (not yet finalized) segment for both tracks and
    /// starts new ones immediately; recording itself is not interrupted.
    func deleteLastSegment() {
        micWriter?.deleteCurrentAndRestart()
        systemWriter?.deleteCurrentAndRestart()
        Log.controller.info("Deleted current in-progress segment for both tracks.")
    }

    // MARK: - Rotation

    private func startRotationTimer() {
        rotationTimer?.invalidate()
        let interval = TimeInterval(settings.rotationMinutes * 60)
        let timer = Timer(timeInterval: interval, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.rotateSegments()
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        rotationTimer = timer
    }

    private func rotateSegments() {
        guard state == .recording else { return }
        micWriter?.rotate()
        systemWriter?.rotate()
        Log.controller.debug("Rotated segments.")
    }

    // MARK: - Recovery

    /// Central "something changed, restart capture" entry point used by
    /// the config-change, default-device-change, and wake handlers below.
    /// Only the capture pipelines are rebuilt -- writers (and thus the
    /// currently open segment files) are left alone.
    private func restartCapture() {
        guard state == .recording else { return }
        Log.controller.notice("Restarting capture pipelines after a device/config change.")
        micCapture?.stop()
        systemCapture?.stop()
        micCapture = nil
        systemCapture = nil
        state = .stopped
        start()
    }

    /// Per spec: "any failure logs via os.Logger and retries after 5s;
    /// never crash silently."
    private func scheduleRetry() {
        restartWorkItem?.cancel()
        let item = DispatchWorkItem { [weak self] in
            self?.start()
        }
        restartWorkItem = item
        DispatchQueue.main.asyncAfter(deadline: .now() + 5, execute: item)
    }

    // MARK: - System observers

    private func registerSystemObservers() {
        // AVAudioEngine configuration changes (e.g. the input device
        // driving the mic tap disappeared/changed sample rate under us).
        configChangeObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in self?.restartCapture() }
        }

        // Sleep/wake: stop cleanly before sleep (finalizing the current
        // segments) rather than leaving a segment spanning the sleep gap,
        // then restart after wake if we were recording before.
        sleepObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.willSleepNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.state == .recording else { return }
                self.wasRecordingBeforeSleep = true
                self.stop()
            }
        }
        wakeObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didWakeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.wasRecordingBeforeSleep else { return }
                self.wasRecordingBeforeSleep = false
                self.start()
            }
        }

        // Default input/output device changes (AirPods connecting,
        // plugging/unplugging headphones or a USB mic, etc).
        //
        // NOTE: verify `AudioObjectPropertyListenerBlock`'s exact closure
        // signature and `AudioObjectAddPropertyListenerBlock`'s parameter
        // order against the real CoreAudio.framework headers -- written
        // here to their commonly documented shape.
        var inputAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var outputAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        let inputBlock: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            Task { @MainActor in self?.restartCapture() }
        }
        let outputBlock: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            Task { @MainActor in self?.restartCapture() }
        }
        defaultInputListenerBlock = inputBlock
        defaultOutputListenerBlock = outputBlock

        _ = AudioObjectAddPropertyListenerBlock(kAudioObjectSystemObject, &inputAddress, DispatchQueue.main, inputBlock)
        _ = AudioObjectAddPropertyListenerBlock(kAudioObjectSystemObject, &outputAddress, DispatchQueue.main, outputBlock)
    }

    // MARK: - Helpers

    /// `{device}` filename component: `Host.current().localizedName`
    /// lower-cased and reduced to ASCII letters/digits/hyphen only (per
    /// spec), e.g. "Sou's MacBook Pro" -> "sou-s-macbook-pro".
    static func sanitizedDeviceID() -> String {
        let raw = Host.current().localizedName ?? "mac"
        let lowered = raw.lowercased()
        var result = ""
        var lastWasDash = false
        for scalar in lowered.unicodeScalars {
            let isLowerAlpha = scalar.value >= 97 && scalar.value <= 122
            let isDigit = scalar.value >= 48 && scalar.value <= 57
            if isLowerAlpha || isDigit {
                result.unicodeScalars.append(scalar)
                lastWasDash = false
            } else if !lastWasDash && !result.isEmpty {
                result.append("-")
                lastWasDash = true
            }
        }
        while result.hasSuffix("-") { result.removeLast() }
        return result.isEmpty ? "mac" : result
    }
}
