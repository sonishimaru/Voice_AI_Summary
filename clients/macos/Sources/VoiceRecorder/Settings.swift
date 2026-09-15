import Foundation
import ServiceManagement

/// Thin `UserDefaults`-backed settings store. Everything is a computed
/// property so `UserDefaults` stays the single source of truth (no
/// in-memory copy that could drift from what's persisted).
final class Settings {
    static let shared = Settings()

    private let defaults = UserDefaults.standard

    private enum Keys {
        static let inboxPath = "inboxPath"
        static let rotationMinutes = "rotationMinutes"
        static let launchAtLogin = "launchAtLogin"
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

    /// Ensures the inbox directory exists, creating intermediate
    /// directories as needed. Safe (and cheap) to call repeatedly.
    func ensureInboxDirectoryExists() throws {
        try FileManager.default.createDirectory(at: inboxURL, withIntermediateDirectories: true)
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
            defaults.set(enabled, forKey: Keys.launchAtLogin)
            return true
        } catch {
            Log.app.error("Failed to \(enabled ? "register" : "unregister", privacy: .public) launch-at-login: \(error.localizedDescription, privacy: .public)")
            return false
        }
    }

    /// Current registration status, read straight from `SMAppService`
    /// (the actual source of truth -- not the mirrored UserDefaults value).
    var isLaunchAtLoginEnabled: Bool {
        SMAppService.mainApp.status == .enabled
    }
}
