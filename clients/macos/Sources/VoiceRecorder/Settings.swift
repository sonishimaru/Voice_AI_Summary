import Foundation
import ServiceManagement

/// The user's *intent* for recording, persisted across launches (see
/// `Settings.intent`). This is deliberately separate from
/// `RecordingController.State`, which reflects whether capture is
/// *actually* running right now (e.g. `state == .stopped` momentarily
/// during a device-change restart even though `intent == .recording`).
enum Intent: String {
    case recording
    case paused
    case stopped
}

/// Thin `UserDefaults`-backed settings store. Everything is a computed
/// property so `UserDefaults` stays the single source of truth (no
/// in-memory copy that could drift from what's persisted).
final class Settings {
    static let shared = Settings()

    private let defaults = UserDefaults.standard

    private enum Keys {
        static let inboxPath = "inboxPath"
        static let rotationMinutes = "rotationMinutes"
        static let recorderIntent = "recorderIntent"
        static let resumeAt = "resumeAt"
        static let stateDirPath = "stateDirPath"
        static let showMenuBarIcon = "showMenuBarIcon"
    }

    private init() {}

    /// Directory the rolling audio files + JSON sidecars are written into.
    /// Defaults to `~/Library/Application Support/VoiceAISummary/inbox/`.
    var inboxURL: URL {
        get {
            if let stored = defaults.string(forKey: Keys.inboxPath), !stored.isEmpty {
                return URL(fileURLWithPath: stored, isDirectory: true)
            }
            return Self.defaultInboxURL
        }
        set { defaults.set(newValue.path, forKey: Keys.inboxPath) }
    }

    static var defaultInboxURL: URL {
        let appSupport = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first ?? URL(fileURLWithPath: NSHomeDirectory() + "/Library/Application Support")
        return appSupport.appendingPathComponent("VoiceAISummary/inbox", isDirectory: true)
    }

    /// Minutes between rolling-file rotations. Default 15 per spec; any
    /// value <= 0 stored in defaults is treated as "unset" and falls back
    /// to the default.
    var rotationMinutes: Int {
        get {
            let stored = defaults.integer(forKey: Keys.rotationMinutes)
            return stored > 0 ? stored : 15
        }
        set { defaults.set(newValue, forKey: Keys.rotationMinutes) }
    }

    /// Whether to also show the menu bar icon. The Dock icon is always
    /// there (the app is not `LSUIElement`), so this one is optional: on a
    /// crowded menu bar -- especially on a notched display, where items
    /// silently disappear under the notch -- it is a small, hard-to-hit
    /// target. Defaults to `true` so nothing vanishes on an existing
    /// install; turn it off from the Dock menu once the Dock icon is
    /// doing the job.
    var showMenuBarIcon: Bool {
        get {
            if defaults.object(forKey: Keys.showMenuBarIcon) == nil { return true }
            return defaults.bool(forKey: Keys.showMenuBarIcon)
        }
        set { defaults.set(newValue, forKey: Keys.showMenuBarIcon) }
    }

    /// Directory `recorder_state.json` / `recorder_events.jsonl` are
    /// written into (see `RecorderStateFile`). Defaults to the inbox
    /// directory's parent, so by default it exists as soon as the inbox
    /// itself has ever been created; overridable via the `stateDirPath`
    /// default for setups that want the state files somewhere else the
    /// ingestion worker watches directly.
    var stateDirectoryURL: URL {
        if let stored = defaults.string(forKey: Keys.stateDirPath), !stored.isEmpty {
            return URL(fileURLWithPath: stored, isDirectory: true)
        }
        return inboxURL.deletingLastPathComponent()
    }

    /// Ensures the inbox directory exists, creating intermediate
    /// directories as needed, and locks it down to owner-only access
    /// (`0700`) since it holds raw microphone/system-audio recordings.
    /// Safe (and cheap) to call repeatedly -- an already-existing
    /// directory just gets its permissions re-asserted.
    func ensureInboxDirectoryExists() throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: inboxURL.path) {
            try fm.setAttributes([.posixPermissions: 0o700], ofItemAtPath: inboxURL.path)
        } else {
            try fm.createDirectory(at: inboxURL, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        }
    }

    // MARK: - Recording intent (persisted, survives relaunch)

    /// The user's last-set recording intent. `nil` means "never set" --
    /// i.e. this is the very first launch ever, which is treated as
    /// `.recording` (today's "always start" behaviour) by
    /// `RecordingController.restoreAtLaunch()`, not by this getter, so
    /// that distinction is still visible to callers that care about it.
    var intent: Intent? {
        get {
            guard let raw = defaults.string(forKey: Keys.recorderIntent) else { return nil }
            return Intent(rawValue: raw)
        }
        set { defaults.set(newValue?.rawValue, forKey: Keys.recorderIntent) }
    }

    /// When a timed pause should automatically resume. `nil` means either
    /// "not paused" or "paused indefinitely" -- `RecordingController`
    /// disambiguates those via `intent`.
    var resumeAt: Date? {
        get {
            let stored = defaults.double(forKey: Keys.resumeAt)
            return stored > 0 ? Date(timeIntervalSince1970: stored) : nil
        }
        set {
            if let newValue {
                defaults.set(newValue.timeIntervalSince1970, forKey: Keys.resumeAt)
            } else {
                defaults.removeObject(forKey: Keys.resumeAt)
            }
        }
    }

    // MARK: - Launch at Login

    /// Registers/unregisters the app as a login item via `SMAppService`.
    ///
    /// NOTE: verify on a real Mac (Xcode 15.3+) that `SMAppService.mainApp`
    /// is reachable from a plain, non-sandboxed app target built by this
    /// xcodegen project without any extra helper-tool bundle, and that no
    /// additional Info.plist entry (beyond what `project.yml` already
    /// generates) is required for `register()`/`unregister()` to succeed.
    @discardableResult
    func setLaunchAtLogin(_ enabled: Bool) -> Bool {
        do {
            if enabled {
                try SMAppService.mainApp.register()
            } else {
                try SMAppService.mainApp.unregister()
            }
            return true
        } catch {
            Log.app.error("Failed to \(enabled ? "register" : "unregister", privacy: .public) launch-at-login: \(error.localizedDescription, privacy: .public)")
            return false
        }
    }

    /// Current registration status, read straight from `SMAppService`
    /// (the actual source of truth -- there is no mirrored UserDefaults
    /// value to drift out of sync with it).
    var isLaunchAtLoginEnabled: Bool {
        SMAppService.mainApp.status == .enabled
    }
}
