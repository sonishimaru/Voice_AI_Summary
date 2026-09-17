import SwiftUI
import AppKit

/// Menubar-only entry point (`LSUIElement = true` in Info.plist means no
/// Dock icon, no regular app menu bar -- just the `MenuBarExtra`).
@main
struct VoiceRecorderApp: App {
    @StateObject private var controller = RecordingController()

    // `NSApplicationDelegateAdaptor` gives us a reliable
    // `applicationDidFinishLaunching` hook to restore recording at launch
    // (per the persisted `intent` -- see `RecordingController.restoreAtLaunch()`).
    // `App.init()` runs before that callback fires, so handing the delegate
    // a reference here is safe.
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    /// Mirrors `Settings.showMenuBarIcon`. `@AppStorage` rather than a read
    /// through `Settings` because SwiftUI has to re-evaluate this scene when
    /// it changes, which is what actually inserts/removes the menu bar item.
    @AppStorage("showMenuBarIcon") private var showMenuBarIcon = true

    init() {
        AppDelegate.controllerForLaunch = controller
    }

    var body: some Scene {
        // The Dock icon is the always-present surface (see `DockIcon`); the
        // menu bar item is opt-out, for menu bars too crowded -- or notches
        // too wide -- to leave room for it.
        MenuBarExtra(isInserted: $showMenuBarIcon) {
            MenuBarContent(controller: controller)
        } label: {
            // NOTE: the icon *shape* (`iconName`) is the real signal for
            // state -- tinting is best-effort in case a colored
            // `Image(systemName:)` doesn't render as intended inside
            // `MenuBarExtra`'s label.
            Image(systemName: iconName)
                .foregroundStyle(iconColor)
        }
        .menuBarExtraStyle(.menu)
    }

    private var iconName: String {
        switch controller.state {
        case .recording: return "waveform.circle.fill"
        case .paused: return "pause.circle.fill"
        case .stopped: return "stop.circle"
        }
    }

    private var iconColor: Color {
        switch controller.state {
        case .recording: return .red
        case .paused: return .orange
        case .stopped: return .gray
        }
    }
}

/// `NSApplicationDelegate` responsible for restoring recording at launch
/// and for making sure a quit (however it's triggered) finalizes the
/// current segment synchronously before the process actually exits.
final class AppDelegate: NSObject, NSApplicationDelegate {
    static var controllerForLaunch: RecordingController?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // `state`'s `didSet` keeps the Dock tile current from here on, but it
        // does not fire for the property's initial value, so paint it once.
        DockIcon.update(for: Self.controllerForLaunch?.state ?? .stopped)
        Self.controllerForLaunch?.restoreAtLaunch()
    }

    /// The Dock icon's right-click (or click-and-hold) menu.
    ///
    /// Carries every action the menu bar item does, because the menu bar
    /// item is optional: on a crowded bar -- or a notched display, where
    /// items disappear entirely -- the Dock tile is a far larger target that
    /// cannot be hidden. Rebuilt on each invocation so it reflects the state
    /// at the moment of the click. macOS appends its own items (Options,
    /// Quit) below these; Quit goes through `applicationShouldTerminate`,
    /// which finalizes the current segment.
    func applicationDockMenu(_ sender: NSApplication) -> NSMenu? {
        guard let controller = Self.controllerForLaunch else { return nil }
        let menu = NSMenu()

        let status = NSMenuItem(title: Self.statusTitle(for: controller), action: nil, keyEquivalent: "")
        status.isEnabled = false
        menu.addItem(status)
        if let message = controller.lastActionMessage {
            let note = NSMenuItem(title: message, action: nil, keyEquivalent: "")
            note.isEnabled = false
            menu.addItem(note)
        }
        menu.addItem(.separator())

        switch controller.state {
        case .recording:
            let pause = NSMenuItem(title: "一時停止", action: nil, keyEquivalent: "")
            let submenu = NSMenu()
            for option in PauseOption.allCases {
                let item = NSMenuItem(title: option.title, action: #selector(dockPause(_:)), keyEquivalent: "")
                item.target = self
                item.tag = option.rawValue
                submenu.addItem(item)
            }
            pause.submenu = submenu
            menu.addItem(pause)
        case .paused:
            menu.addItem(Self.item(title: "再開", action: #selector(dockResume(_:)), target: self))
        case .stopped:
            menu.addItem(Self.item(title: "開始", action: #selector(dockStart(_:)), target: self))
        }

        let stop = Self.item(title: "停止", action: #selector(dockStop(_:)), target: self)
        // An already-stopped recorder has nothing to stop; `isEnabled` has to
        // be set explicitly because a menu built outside the responder chain
        // does not get automatic enabling.
        stop.isEnabled = controller.intent != .stopped
        menu.addItem(stop)

        menu.addItem(.separator())
        menu.addItem(Self.item(title: "直近 15 分の録音を削除…", action: #selector(dockDeleteRecent(_:)), target: self))
        menu.addItem(Self.item(title: "inbox フォルダを開く", action: #selector(dockOpenInbox(_:)), target: self))

        let menuBarToggle = Self.item(title: "メニューバーに表示", action: #selector(dockToggleMenuBarIcon(_:)), target: self)
        menuBarToggle.state = Settings.shared.showMenuBarIcon ? .on : .off
        menu.addItem(menuBarToggle)

        return menu
    }

    private static func item(title: String, action: Selector, target: AnyObject) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: action, keyEquivalent: "")
        item.target = target
        return item
    }

    private static func statusTitle(for controller: RecordingController) -> String {
        switch controller.state {
        case .recording:
            guard let startedAt = controller.recordingStartedAt else { return "録音中" }
            return "録音中 \(timeFormatter.string(from: startedAt))〜"
        case .paused:
            guard let resumeAt = controller.resumeAt else { return "一時停止中 — 手動で再開するまで" }
            let remaining = max(0, Int(resumeAt.timeIntervalSinceNow / 60))
            return "一時停止中 — \(timeFormatter.string(from: resumeAt)) に再開（残り \(remaining) 分）"
        case .stopped:
            return controller.intent == .recording ? "録音を開始できません — 5 秒ごとに再試行中" : "停止中"
        }
    }

    private static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm"
        return formatter
    }()

    // MARK: - Dock menu actions

    @objc private func dockPause(_ sender: NSMenuItem) {
        guard let option = PauseOption(rawValue: sender.tag) else { return }
        Self.controllerForLaunch?.pause(until: option.resumeDate())
    }

    @objc private func dockResume(_ sender: NSMenuItem) {
        Self.controllerForLaunch?.resume(reason: "user")
    }

    @objc private func dockStart(_ sender: NSMenuItem) {
        Self.controllerForLaunch?.start()
    }

    @objc private func dockStop(_ sender: NSMenuItem) {
        Self.controllerForLaunch?.stop()
    }

    @objc private func dockDeleteRecent(_ sender: NSMenuItem) {
        Self.controllerForLaunch?.deleteRecentAudio()
    }

    @objc private func dockOpenInbox(_ sender: NSMenuItem) {
        try? Settings.shared.ensureInboxDirectoryExists()
        NSWorkspace.shared.open(Settings.shared.inboxURL)
    }

    @objc private func dockToggleMenuBarIcon(_ sender: NSMenuItem) {
        Settings.shared.showMenuBarIcon.toggle()
    }

    /// Covers logout/shutdown (and any other path that asks the app to
    /// terminate through this callback rather than a direct
    /// `terminate(_:)` call): finalize synchronously, then allow
    /// termination to proceed.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        Self.controllerForLaunch?.prepareForQuit()
        return .terminateNow
    }

    /// Belt-and-suspenders for any termination path that reaches this
    /// callback without having gone through `applicationShouldTerminate`
    /// first; `prepareForQuit()` is idempotent, so calling it twice is
    /// harmless.
    func applicationWillTerminate(_ notification: Notification) {
        Self.controllerForLaunch?.prepareForQuit()
    }
}

/// The `MenuBarExtra`'s dropdown contents.
private struct MenuBarContent: View {
    @ObservedObject var controller: RecordingController

    var body: some View {
        statusLine
        if let message = controller.lastActionMessage {
            Text(message)
        }

        switch controller.state {
        case .recording:
            Menu("一時停止") {
                ForEach(PauseOption.allCases, id: \.rawValue) { option in
                    Button(option.title) { controller.pause(until: option.resumeDate()) }
                }
            }
        case .paused:
            Button("再開") { controller.resume(reason: "user") }
        case .stopped:
            Button("開始") { controller.start() }
        }

        Button("停止") { controller.stop() }
            .disabled(controller.intent == .stopped)

        Divider()

        Button("直近 15 分の録音を削除…") {
            controller.deleteRecentAudio()
        }

        Divider()

        Toggle("メニューバーに表示", isOn: menuBarIconBinding)
        Toggle("ログイン時に起動", isOn: launchAtLoginBinding)
        Button("inbox フォルダを開く") {
            openInboxFolder()
        }

        Divider()

        Button("終了") {
            controller.prepareForQuit()
            NSApplication.shared.terminate(nil)
        }
    }

    @ViewBuilder
    private var statusLine: some View {
        switch controller.state {
        case .recording:
            if let startedAt = controller.recordingStartedAt {
                Text("録音中 \(Self.timeFormatter.string(from: startedAt))〜")
            } else {
                Text("録音中")
            }
        case .paused:
            Text(pausedStatusText)
        case .stopped:
            if controller.intent == .recording {
                Text("録音を開始できません — 5 秒ごとに再試行中")
            } else {
                Text("停止中")
            }
        }
    }

    /// "一時停止中 — 10:30 に再開（残り 28 分）" for a timed pause, or
    /// "一時停止中 — 手動で再開するまで" for an indefinite one. Reads
    /// `controller.now` (not `Date()`) so the "残り" countdown is tied to
    /// the controller's own 30s ticker rather than to whenever SwiftUI
    /// happens to re-evaluate the view for some unrelated reason.
    private var pausedStatusText: String {
        guard let resumeAt = controller.resumeAt else {
            return "一時停止中 — 手動で再開するまで"
        }
        let remainingMinutes = max(0, Int(resumeAt.timeIntervalSince(controller.now) / 60))
        return "一時停止中 — \(Self.timeFormatter.string(from: resumeAt)) に再開（残り \(remainingMinutes) 分）"
    }

    /// Turning this off leaves the Dock icon as the only way in -- which is
    /// the point, and why the Dock menu carries every action this one does.
    private var menuBarIconBinding: Binding<Bool> {
        Binding(
            get: { Settings.shared.showMenuBarIcon },
            set: { Settings.shared.showMenuBarIcon = $0 }
        )
    }

    /// `Settings` isn't `ObservableObject`; a hand-rolled `Binding` reads
    /// the live `SMAppService` status on `get` and pushes changes through
    /// `Settings.setLaunchAtLogin` on `set`, reverting the toggle in the UI
    /// if registration actually failed.
    private var launchAtLoginBinding: Binding<Bool> {
        Binding(
            get: { Settings.shared.isLaunchAtLoginEnabled },
            set: { newValue in
                _ = Settings.shared.setLaunchAtLogin(newValue)
            }
        )
    }

    private func openInboxFolder() {
        try? Settings.shared.ensureInboxDirectoryExists()
        NSWorkspace.shared.open(Settings.shared.inboxURL)
    }

    private static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm"
        return formatter
    }()
}
