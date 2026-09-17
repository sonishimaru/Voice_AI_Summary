import Foundation
import Combine
import AVFoundation
import AudioToolbox
import AppKit
import UserNotifications

/// Top-level state machine for the app: owns both capture pipelines and
/// their writers, the rotation timer, pause/resume, the "delete recent
/// audio" privacy control, and recovery from device/sleep/wake changes.
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
    // conformance) because `startCapture`/`pause`/`resume`/the recovery
    // paths, and the SwiftUI menu content, all compare `state` with `==`.
    //
    // `state` reflects whether capture is *actually* running right now.
    // It is deliberately separate from `intent` (see `Settings.Intent`
    // and the rule above `startCapture` below): `state` can be
    // `.stopped` for a few hundred milliseconds during a device-change
    // restart even though `intent == .recording` the whole time.
    enum State: Equatable {
        case recording
        case paused
        case stopped
    }

    @Published private(set) var state: State = .stopped {
        // The Dock tile is this app's only always-visible surface (the menu
        // bar icon is optional, and can be hidden under the notch), so it
        // follows `state` from the one place every transition goes through.
        didSet { DockIcon.update(for: state) }
    }
    @Published private(set) var recordingStartedAt: Date?
    @Published var resumeAt: Date? {
        didSet { settings.resumeAt = resumeAt }
    }
    /// Ticks every 30s while paused with a scheduled auto-resume, purely
    /// so the "残り N 分" countdown in the menu re-renders periodically;
    /// nothing reads its value for any other purpose.
    @Published var now: Date = Date()
    /// Set by `deleteRecentAudio` to surface a one-line confirmation in
    /// the menu; cleared by every other user action.
    @Published var lastActionMessage: String?

    /// The user's persisted intent -- the only thing that decides whether
    /// an automatic restart path is allowed to actually start capturing.
    /// Mirrored to `Settings` on every change so it survives a relaunch.
    /// `@Published` so the menu can key "停止" on it: while capture is
    /// failing and retrying, `state` is `.stopped` but `intent` is still
    /// `.recording`, and the user must still be able to stop the retries.
    @Published private(set) var intent: Intent = .stopped {
        didSet { settings.intent = intent }
    }

    /// When the *current* `state` began; feeds the `since` field of
    /// `recorder_state.json`.
    private var stateSince = Date()

    private let settings = Settings.shared
    private let stateFile = RecorderStateFile.shared

    private var micCapture: MicCapture?
    private var systemCapture: ProcessTapCapture?

    // Writers now live for exactly one capture session: every teardown
    // path (pause/stop/sleep/quit) finalizes and nils both writers, and
    // `startCapture` always creates fresh ones when they're nil. This is
    // what makes pause/stop trustworthy -- there is never a `.part` file
    // left open across a pause, and pre-/post-pause audio can never be
    // spliced into one file with no marker between them. (Only
    // `restartCapture`, the device/config-change recovery path below,
    // intentionally reuses the still-open writers -- it never finalizes
    // anything, since a brief device hiccup shouldn't fragment the
    // recording into extra tiny files.)
    private var micWriter: SegmentWriter?
    private var systemWriter: SegmentWriter?

    private var rotationTimer: Timer?
    private var restartWorkItem: DispatchWorkItem?
    /// Fires once, at `resumeAt`, to auto-resume a timed pause.
    private var resumeTimer: Timer?
    /// Rewrites `recorder_state.json` every 60s while recording, so a
    /// reader never has to guess whether a missing update means "stopped"
    /// or "the app just hasn't had anything else to report."
    private var heartbeatTimer: Timer?
    /// Drives `now` every 30s while paused with a scheduled auto-resume.
    private var nowTicker: Timer?

    private var hasPreparedForQuit = false
    private var notificationAuthorizationRequested = false

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

    // MARK: - Public controls (user actions)

    /// User-initiated start. Equivalent to `resume(reason: "user")` --
    /// there is no meaningful difference between "start from stopped" and
    /// "resume from paused" once writers are always session-scoped.
    func start() {
        resume(reason: "user")
    }

    /// Pauses: finalizes the current segment (closes + renames both
    /// writers' `.part` files) and tears down capture. If `until` is nil,
    /// stays paused until the user explicitly resumes; otherwise persists
    /// `until` and schedules an automatic resume there (surviving a
    /// relaunch in the meantime, via `restoreAtLaunch`).
    func pause(until: Date?) {
        // Keyed on `intent`, not `state`: while capture is failing and the
        // 5 s retry loop is running, `state` is `.stopped` but `intent` is
        // `.recording` -- and that is exactly when the user needs Pause to
        // work, or the retries keep firing. `tearDown` is safe with no
        // writers open.
        guard intent == .recording else { return }
        intent = .paused
        lastActionMessage = nil

        tearDown(finalize: true)
        state = .paused
        stateSince = Date()
        resumeAt = until

        if until != nil {
            requestNotificationAuthorizationIfNeeded()
            startNowTicker()
        }
        scheduleResumeTimer(for: until)
        writeStateFile(reason: "user")
        Log.controller.info("Paused (resumeAt=\(until.map { String(describing: $0) } ?? "nil", privacy: .public)).")
    }

    /// Resumes recording -- used for the user's "再開"/"開始" menu actions
    /// (`reason: "user"`) and for an elapsed timed pause (`reason:
    /// "timer"`), which is treated as a user-equivalent action: an auto-
    /// resume is exactly what the user asked to happen when they set the
    /// timer. This is the only other place (besides `start()`, which just
    /// calls this) allowed to set `intent = .recording`.
    func resume(reason: String) {
        intent = .recording
        resumeTimer?.invalidate()
        resumeTimer = nil
        stopNowTicker()
        resumeAt = nil
        lastActionMessage = nil
        startCapture(reason: reason)
        Log.controller.info("Resumed (\(reason, privacy: .public)).")
    }

    /// Fully stops: finalizes the current segment and stays stopped until
    /// the user presses "開始" again (or relaunches while `intent ==
    /// .recording`).
    func stop() {
        guard intent != .stopped else { return }  // same reasoning as in `pause`
        intent = .stopped
        resumeTimer?.invalidate()
        resumeTimer = nil
        stopNowTicker()
        resumeAt = nil
        lastActionMessage = nil

        tearDown(finalize: true)
        state = .stopped
        stateSince = Date()
        writeStateFile(reason: "user")
        Log.controller.info("Stopped by user.")
    }

    /// "Delete recent audio" -- the privacy control for "I said something
    /// I shouldn't have, or I'm about to." Confirms first (destructive,
    /// and only partial -- see the alert text), then discards the
    /// currently-open segment (if recording) and sweeps the inbox for
    /// already-rotated files young enough to overlap the window.
    func deleteRecentAudio(minutes: Int = 15) {
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = "直近 \(minutes) 分の録音を削除しますか？"
        alert.informativeText = "現在録音中のセグメントと、直近 \(minutes) 分以内に開始されたインボックス内のファイルを削除します。すでに Claude Desktop に取り込まれた分がある場合、この操作だけでは消えません。取り込み済みの音声は Claude Desktop 側で delete_range を使って削除してください。"
        alert.addButton(withTitle: "削除")
        alert.addButton(withTitle: "キャンセル")

        guard alert.runModal() == .alertFirstButtonReturn else {
            Log.controller.info("Delete recent audio cancelled by user.")
            return
        }

        if state == .recording {
            micWriter?.deleteCurrentAndRestart()
            systemWriter?.deleteCurrentAndRestart()
        }

        let deletedFiles = deleteRotatedInboxFiles(newerThanMinutesAgo: minutes)
        lastActionMessage = "削除: 現在のセグメント + \(deletedFiles) ファイル（取り込み済みの分は Claude Desktop の delete_range で）"
        stateFile.appendEventOnly(makeSnapshot(reason: "delete_recent", minutes: minutes, files: deletedFiles))
        Log.controller.info("Deleted recent audio: current segment + \(deletedFiles) rotated file(s), window=\(minutes)m.")
    }

    // MARK: - Lifecycle entry points (called from VoiceRecorderApp/AppDelegate)

    /// Called once from `applicationDidFinishLaunching`. Restores whatever
    /// `intent` was persisted from the previous run instead of
    /// unconditionally starting -- this is what makes a pause survive a
    /// quit+relaunch. `intent == nil` (the key has never been written)
    /// means this is the very first launch ever, which is treated as
    /// `.recording` -- today's "always start" default.
    func restoreAtLaunch() {
        stateSince = Date()
        intent = settings.intent ?? .recording
        resumeAt = settings.resumeAt

        switch intent {
        case .recording:
            startCapture(reason: "launch")
            if state != .recording {
                // `startCapture` scheduled a retry instead of succeeding;
                // write the state file now so a reader sees the fresh pid
                // promptly rather than waiting up to 5s for the retry.
                writeStateFile(reason: "launch")
            }
        case .paused:
            state = .paused
            if let resumeAt, resumeAt <= Date() {
                resume(reason: "timer")
                postAutoResumeNotification()
            } else if let resumeAt {
                scheduleResumeTimer(for: resumeAt)
                startNowTicker()  // the countdown in the status line
                writeStateFile(reason: "launch")
            } else {
                writeStateFile(reason: "launch")
            }
        case .stopped:
            state = .stopped
            writeStateFile(reason: "launch")
        }
        Log.controller.info("Restored at launch: intent=\(self.intent.rawValue, privacy: .public).")
    }

    /// Synchronous teardown for process termination (終了 menu item,
    /// `applicationWillTerminate`, `applicationShouldTerminate`):
    /// finalizes both writers *synchronously* via `SegmentWriter.closeSync()`
    /// and writes the state file synchronously too, because none of the
    /// usual `queue.async` dispatches this controller relies on elsewhere
    /// are guaranteed to run once the process starts tearing down -- an
    /// `async` close here could easily lose the rename of the `.part` file
    /// to `.m4a`. Idempotent: safe to call from more than one termination
    /// hook without double-finalizing.
    func prepareForQuit() {
        guard !hasPreparedForQuit else { return }
        hasPreparedForQuit = true

        rotationTimer?.invalidate()
        rotationTimer = nil
        heartbeatTimer?.invalidate()
        heartbeatTimer = nil
        resumeTimer?.invalidate()
        resumeTimer = nil
        restartWorkItem?.cancel()
        restartWorkItem = nil
        stopNowTicker()

        micCapture?.stop()
        systemCapture?.stop()
        micCapture = nil
        systemCapture = nil

        micWriter?.closeSync()
        systemWriter?.closeSync()
        micWriter = nil
        systemWriter = nil

        state = .stopped
        stateSince = Date()
        stateFile.writeSync(makeSnapshot(reason: "quit"))
        Log.controller.info("Prepared for quit: finalized writers synchronously.")
    }

    // MARK: - Capture (the trustworthy-pause rule)

    /// THE RULE: every *automatic* (non-user-initiated) path that might
    /// restart capture calls `startCapture(reason:)`, never `start()` or
    /// `resume()` directly -- and `startCapture` itself begins with
    /// `guard intent == .recording else { return }`. Only the two
    /// user-facing actions, `start()`/`resume(reason:)` (identical) and
    /// the paths that are explicitly treated as user-equivalent, are
    /// allowed to set `intent = .recording` at all. That is the whole
    /// mechanism that makes Pause trustworthy: nothing automatic can ever
    /// flip a paused/stopped app back into recording, because none of
    /// the automatic paths touch `intent`, and all of them are gated on
    /// it already being `.recording`.
    ///
    /// The automatic call sites, and why each is safe:
    ///   1. `restoreAtLaunch()` (reason "launch") -- only calls this when
    ///      `intent == .recording` was the persisted value; a paused or
    ///      stopped app relaunches paused or stopped.
    ///   2. `scheduleRetry()`'s work item (reason "retry") -- re-checks
    ///      `intent` itself 5s later; a Pause/Stop issued in the meantime
    ///      (which cancels `restartWorkItem`) or that already flipped
    ///      `intent` makes the retry a no-op either way.
    ///   3. `restartCapture()` (reason "device-change") -- guarded on
    ///      `state == .recording`, which can only be true if `intent` was
    ///      already `.recording`; never touches `intent` itself.
    ///   4. `didWake()` (reason "wake") -- only calls this when `intent ==
    ///      .recording`.
    /// `resumeTimer`'s fire handler (`resumeTimerFired`) is deliberately
    /// *not* on this list: an elapsed timed pause calls `resume(reason:
    /// "timer")`, the user-equivalent path, because that is exactly what
    /// the user asked to happen when they set the timer.
    private func startCapture(reason: String) {
        guard intent == .recording else {
            Log.controller.info("startCapture(\(reason, privacy: .public)) skipped: intent is \(self.intent.rawValue, privacy: .public).")
            return
        }
        guard state != .recording else { return }

        do {
            try settings.ensureInboxDirectoryExists()
        } catch {
            Log.controller.error("Could not create inbox directory: \(error.localizedDescription, privacy: .public)")
            scheduleRetry()
            return
        }

        // Writers are nil after any teardown (pause/stop/sleep/quit all
        // finalize them), so a fresh capture session always gets fresh
        // writers. `restartCapture` (device/config change) is the one
        // path that reaches here with non-nil writers already open, and
        // this correctly leaves them alone.
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
        stateSince = recordingStartedAt ?? Date()
        stopNowTicker()
        startRotationTimer()
        startHeartbeat()
        writeStateFile(reason: reason)
        Log.controller.info("Capture started (\(reason, privacy: .public)).")
    }

    /// Tears down both capture pipelines and (when `finalize`) finalizes
    /// (closes + renames) both writers. Every user/lifecycle teardown
    /// path -- pause, stop, sleep, quit -- calls this with `finalize:
    /// true`, which is what guarantees a `.part` file is never abandoned
    /// and pre-/post-teardown audio is never spliced into one file.
    private func tearDown(finalize: Bool) {
        rotationTimer?.invalidate()
        rotationTimer = nil
        heartbeatTimer?.invalidate()
        heartbeatTimer = nil
        restartWorkItem?.cancel()
        restartWorkItem = nil
        stopNowTicker()

        micCapture?.stop()
        systemCapture?.stop()
        micCapture = nil
        systemCapture = nil

        if finalize {
            micWriter?.close()
            systemWriter?.close()
        }
        micWriter = nil
        systemWriter = nil

        state = .stopped
        recordingStartedAt = nil
        stateSince = Date()
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

    private func startHeartbeat() {
        heartbeatTimer?.invalidate()
        let timer = Timer(timeInterval: 60, repeats: true) { [weak self] _ in
            Task { @MainActor in
                // State file only: a heartbeat is not a transition and must
                // not land in `recorder_events.jsonl`.
                guard let self else { return }
                self.stateFile.writeStateOnly(self.makeSnapshot(reason: "heartbeat"))
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        heartbeatTimer = timer
    }

    // MARK: - Recovery

    /// Central "something changed, restart capture" entry point used by
    /// the config-change and default-device-change handlers below. Only
    /// the capture pipelines are rebuilt -- writers (and thus the
    /// currently open segment files) are left alone, and `intent` is
    /// never touched (see the rule above `startCapture`).
    private func restartCapture() {
        guard state == .recording else { return }
        Log.controller.notice("Restarting capture pipelines after a device/config change.")
        micCapture?.stop()
        systemCapture?.stop()
        micCapture = nil
        systemCapture = nil
        state = .stopped
        startCapture(reason: "device-change")
    }

    /// Per spec: "any failure logs via os.Logger and retries after 5s;
    /// never crash silently." `pause()`/`stop()` cancel this work item
    /// via `tearDown`, and `startCapture` re-checks `intent` when it
    /// eventually runs, so a Pause/Stop issued during the 5s window is
    /// never silently overridden by the retry.
    private func scheduleRetry() {
        restartWorkItem?.cancel()
        let item = DispatchWorkItem { [weak self] in
            self?.startCapture(reason: "retry")
        }
        restartWorkItem = item
        DispatchQueue.main.asyncAfter(deadline: .now() + 5, execute: item)
    }

    // MARK: - Timed pause

    /// Schedules (or, given `nil`, cancels) `resumeTimer` to fire at
    /// `date`. Does not itself decide past-vs-future -- callers that need
    /// "resume immediately if `date` is already past" (`restoreAtLaunch`,
    /// `didWake`) check that themselves before deciding whether to call
    /// this or `resume(reason:)` directly.
    private func scheduleResumeTimer(for date: Date?) {
        resumeTimer?.invalidate()
        resumeTimer = nil
        guard let date else { return }
        let interval = max(date.timeIntervalSinceNow, 0.1)
        let timer = Timer(timeInterval: interval, repeats: false) { [weak self] _ in
            Task { @MainActor in
                self?.resumeTimerFired()
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        resumeTimer = timer
    }

    private func resumeTimerFired() {
        guard intent == .paused, let resumeAt, resumeAt <= Date() else { return }
        resume(reason: "timer")
        postAutoResumeNotification()
    }

    private func startNowTicker() {
        stopNowTicker()
        now = Date()
        let timer = Timer(timeInterval: 30, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.now = Date()
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        nowTicker = timer
    }

    private func stopNowTicker() {
        nowTicker?.invalidate()
        nowTicker = nil
    }

    /// Requests `.alert` notification authorization the first time a
    /// *timed* pause is set (not an indefinite one, since there is then
    /// nothing to notify about), rather than at the moment the timer
    /// actually fires -- which could be while the user is away, when a
    /// permission prompt would be useless. Calling `requestAuthorization`
    /// again after the user has already answered is harmless (it just
    /// reports the existing decision without re-prompting), so the
    /// `notificationAuthorizationRequested` guard here is purely to avoid
    /// asking on every single timed pause in one session.
    ///
    /// NOTE: verify on a real Mac that a plain, code-signed (even ad-hoc)
    /// `LSUIElement` app can call `UNUserNotificationCenter` at all
    /// without further Info.plist/entitlement setup, and that a posted
    /// notification actually shows a banner without a
    /// `UNUserNotificationCenterDelegate` being set (none is set here).
    private func requestNotificationAuthorizationIfNeeded() {
        guard !notificationAuthorizationRequested else { return }
        notificationAuthorizationRequested = true
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert]) { granted, error in
            if let error {
                Log.controller.error("Notification authorization request failed: \(error.localizedDescription, privacy: .public)")
            } else {
                Log.controller.info("Notification authorization granted: \(granted, privacy: .public).")
            }
        }
    }

    /// Posts "録音を再開しました" so a paused user finds out an automatic
    /// resume happened without having to notice the menu-bar icon
    /// changed. Falls back to `osascript` if `UNUserNotificationCenter`
    /// authorization was denied, or posting through it fails outright.
    ///
    /// The delivery helpers below are `nonisolated`: `UNUserNotificationCenter`
    /// invokes its completion handlers on an arbitrary background queue
    /// (per Apple's docs), never necessarily the main actor, and none of
    /// these helpers touch any `@MainActor`-isolated controller state --
    /// only `Log` (thread-safe `os.Logger`s) and standalone system objects
    /// -- so there is nothing that actually needs the hop back to main.
    private func postAutoResumeNotification() {
        let center = UNUserNotificationCenter.current()
        center.getNotificationSettings { settings in
            switch settings.authorizationStatus {
            case .authorized, .provisional:
                Self.deliverViaUserNotificationCenter()
            case .notDetermined:
                center.requestAuthorization(options: [.alert]) { granted, _ in
                    if granted {
                        Self.deliverViaUserNotificationCenter()
                    } else {
                        Self.deliverViaOsascript()
                    }
                }
            default:
                Self.deliverViaOsascript()
            }
        }
    }

    private nonisolated static func deliverViaUserNotificationCenter() {
        let content = UNMutableNotificationContent()
        content.title = "VoiceRecorder"
        content.body = "録音を再開しました"
        let request = UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil)
        UNUserNotificationCenter.current().add(request) { error in
            if let error {
                Log.controller.error("Failed to post local notification: \(error.localizedDescription, privacy: .public)")
                Self.deliverViaOsascript()
            }
        }
    }

    private nonisolated static func deliverViaOsascript() {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        process.arguments = ["-e", "display notification \"録音を再開しました\" with title \"VoiceRecorder\""]
        do {
            try process.run()
        } catch {
            Log.controller.error("Failed to post fallback notification via osascript: \(error.localizedDescription, privacy: .public)")
        }
    }

    // MARK: - Delete recent audio

    /// Sweeps `settings.inboxURL` for already-rotated `mac_mic_*`/
    /// `mac_system_*` `.m4a` files whose *rotation window* overlaps the
    /// last `minutes` minutes -- i.e. `startedAt + rotationMinutes*60 >
    /// now - minutes*60` -- so a file that finished rotating just barely
    /// outside a naive "started in the last N minutes" check, but whose
    /// *content* still falls inside the window, is still caught. Returns
    /// the number of `.m4a` files removed (sidecars are removed too but
    /// not counted).
    private func deleteRotatedInboxFiles(newerThanMinutesAgo minutes: Int) -> Int {
        let fm = FileManager.default
        let inbox = settings.inboxURL
        let cutoff = Date().addingTimeInterval(-Double(minutes) * 60)
        let rotationSeconds = Double(settings.rotationMinutes * 60)

        let entries: [URL]
        do {
            entries = try fm.contentsOfDirectory(at: inbox, includingPropertiesForKeys: nil)
        } catch {
            Log.controller.error("Failed to list inbox for delete-recent sweep: \(error.localizedDescription, privacy: .public)")
            return 0
        }

        var deleted = 0
        for url in entries {
            let name = url.lastPathComponent
            guard name.hasSuffix(".m4a"),
                  name.hasPrefix(AudioSource.mic.rawValue + "_") || name.hasPrefix(AudioSource.system.rawValue + "_"),
                  let startedAt = Self.startedAt(fromRotatedFilename: name)
            else { continue }
            guard startedAt.addingTimeInterval(rotationSeconds) > cutoff else { continue }

            do {
                try fm.removeItem(at: url)
                deleted += 1
            } catch {
                Log.controller.error("Failed to delete \(name, privacy: .public): \(error.localizedDescription, privacy: .public)")
                continue
            }

            let sidecarURL = url.deletingPathExtension().appendingPathExtension("json")
            if fm.fileExists(atPath: sidecarURL.path) {
                do {
                    try fm.removeItem(at: sidecarURL)
                } catch {
                    Log.controller.error("Failed to delete sidecar for \(name, privacy: .public): \(error.localizedDescription, privacy: .public)")
                }
            }
        }
        return deleted
    }

    /// Parses the `{YYYYMMDDTHHMMSSZ}` timestamp out of a
    /// `{source}_{device}_{timestamp}.m4a` filename. The device component
    /// and the timestamp are both underscore-free (`sanitizedDeviceID()`
    /// only ever emits lowercase letters/digits/hyphens; the timestamp is
    /// digits plus `T`/`Z`), so splitting on the *last* underscore
    /// isolates the timestamp correctly regardless of how many
    /// underscores appear earlier in `{source}` (`mac_mic`/`mac_system`
    /// each contain one).
    private static func startedAt(fromRotatedFilename name: String) -> Date? {
        let stem = (name as NSString).deletingPathExtension
        guard let lastUnderscore = stem.lastIndex(of: "_") else { return nil }
        let timestamp = String(stem[stem.index(after: lastUnderscore)...])
        return SegmentWriter.filenameTimestampFormatter.date(from: timestamp)
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

        // Sleep/wake: finalize cleanly before sleep (rather than leaving a
        // segment spanning the sleep gap) and pick back up after wake
        // based on `intent`/`resumeAt`, not a one-off flag.
        sleepObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.willSleepNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in self?.willSleep() }
        }
        wakeObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didWakeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in self?.didWake() }
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

        _ = AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &inputAddress, DispatchQueue.main, inputBlock)
        _ = AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &outputAddress, DispatchQueue.main, outputBlock)
    }

    /// Runs before sleep. If actually recording, finalize now rather than
    /// leaving a segment spanning the sleep gap -- `intent` is left
    /// untouched (still `.recording`), so this is not a pause; `didWake`
    /// (or even a relaunch during sleep) restarts capture automatically.
    /// If paused with a scheduled auto-resume, invalidate the timer (it
    /// may not fire reliably across sleep anyway) -- `didWake` recomputes
    /// past-vs-future from `resumeAt` and reschedules as needed.
    private func willSleep() {
        if state == .recording {
            tearDown(finalize: true)
            state = .stopped
            stateSince = Date()
            writeStateFile(reason: "sleep")
        } else if state == .paused {
            resumeTimer?.invalidate()
            resumeTimer = nil
            stopNowTicker()
        }
        Log.controller.info("Going to sleep; intent=\(self.intent.rawValue, privacy: .public).")
    }

    /// Runs after wake, purely from `intent`/`resumeAt` -- there is no
    /// "was recording before sleep" flag to fall out of sync with reality.
    private func didWake() {
        switch intent {
        case .recording:
            startCapture(reason: "wake")
        case .paused:
            if let resumeAt, resumeAt <= Date() {
                resume(reason: "timer")
                postAutoResumeNotification()
            } else if resumeAt != nil {
                scheduleResumeTimer(for: resumeAt)
                startNowTicker()  // `willSleep` stopped it; the countdown must move again
            }
        case .stopped:
            break
        }
        Log.controller.info("Woke from sleep; intent=\(self.intent.rawValue, privacy: .public).")
    }

    // MARK: - State file

    private func stateFileStateString() -> String {
        switch state {
        case .recording: return "recording"
        case .paused: return "paused"
        case .stopped: return "stopped"
        }
    }

    private func makeSnapshot(reason: String, minutes: Int? = nil, files: Int? = nil) -> RecorderStateSnapshot {
        RecorderStateSnapshot(
            state: stateFileStateString(),
            since: RecorderStateFile.iso8601UTCFormatter.string(from: stateSince),
            resumeAt: resumeAt.map { RecorderStateFile.iso8601UTCFormatter.string(from: $0) },
            reason: reason,
            pid: ProcessInfo.processInfo.processIdentifier,
            appVersion: RecorderStateFile.appVersion,
            updatedAt: RecorderStateFile.iso8601UTCFormatter.string(from: Date()),
            minutes: minutes,
            files: files
        )
    }

    private func writeStateFile(reason: String) {
        stateFile.write(makeSnapshot(reason: reason))
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
