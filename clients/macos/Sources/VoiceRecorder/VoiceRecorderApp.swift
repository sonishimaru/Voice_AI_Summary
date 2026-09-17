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

    init() {
        AppDelegate.controllerForLaunch = controller
    }

    var body: some Scene {
        MenuBarExtra {
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
        Self.controllerForLaunch?.restoreAtLaunch()
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
                Button("30 分") { controller.pause(until: Date().addingTimeInterval(30 * 60)) }
                Button("1 時間") { controller.pause(until: Date().addingTimeInterval(60 * 60)) }
                Button("今日中") { controller.pause(until: Self.endOfToday()) }
                Button("再開するまで") { controller.pause(until: nil) }
            }
        case .paused:
            Button("再開") { controller.resume(reason: "user") }
        case .stopped:
            Button("開始") { controller.start() }
        }

        Button("停止") { controller.stop() }
            .disabled(controller.state == .stopped)

        Divider()

        Button("直近 15 分の録音を削除…") {
            controller.deleteRecentAudio()
        }

        Divider()

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
            Text("停止中")
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

    /// Next local midnight, for the "今日中" (until end of today) pause option.
    private static func endOfToday() -> Date {
        Calendar.current.nextDate(after: Date(), matching: DateComponents(hour: 0, minute: 0), matchingPolicy: .nextTime)
            ?? Date().addingTimeInterval(24 * 60 * 60)
    }

    private static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm"
        return formatter
    }()
}
