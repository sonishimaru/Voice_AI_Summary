import SwiftUI
import AppKit

/// Menubar-only entry point (`LSUIElement = true` in Info.plist means no
/// Dock icon, no regular app menu bar -- just the `MenuBarExtra`).
@main
struct VoiceRecorderApp: App {
    @StateObject private var controller = RecordingController()

    // `NSApplicationDelegateAdaptor` gives us a reliable
    // `applicationDidFinishLaunching` hook to auto-start recording at
    // launch (per spec). `App.init()` runs before that callback fires, so
    // handing the delegate a reference here is safe.
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    init() {
        AppDelegate.controllerForLaunch = controller
    }

    var body: some Scene {
        MenuBarExtra {
            MenuBarContent(controller: controller)
        } label: {
            // NOTE: verify on a real Mac that a colored (non-template)
            // `Image(systemName:)` actually renders with color in the menu
            // bar via `MenuBarExtra`'s default label -- some SwiftUI/AppKit
            // combinations force menu-bar status images to render as
            // monochrome "template" images regardless of `foregroundStyle`.
            // If tinting doesn't show up, drive an `NSStatusItem` image
            // directly (with `isTemplate = false`) instead of SwiftUI's
            // `MenuBarExtra` label.
            Image(systemName: iconName)
                .foregroundStyle(iconColor)
        }
        .menuBarExtraStyle(.menu)
    }

    private var iconName: String {
        switch controller.state {
        case .recording: return "waveform.circle.fill"
        case .paused, .stopped: return "waveform.circle"
        }
    }

    private var iconColor: Color {
        controller.state == .recording ? .red : .gray
    }
}

/// Minimal `NSApplicationDelegate` whose only job is to auto-start
/// recording once the app has actually finished launching.
final class AppDelegate: NSObject, NSApplicationDelegate {
    static var controllerForLaunch: RecordingController?

    func applicationDidFinishLaunching(_ notification: Notification) {
        Self.controllerForLaunch?.start()
    }
}

/// The `MenuBarExtra`'s dropdown contents.
private struct MenuBarContent: View {
    @ObservedObject var controller: RecordingController

    var body: some View {
        statusLine

        Divider()

        pauseResumeButton
        Button("Delete last 15 minutes") {
            controller.deleteLastSegment()
        }

        Divider()

        Toggle("Launch at Login", isOn: launchAtLoginBinding)
        Button("Open inbox folder") {
            openInboxFolder()
        }

        Divider()

        Button("Quit") {
            NSApplication.shared.terminate(nil)
        }
    }

    @ViewBuilder
    private var statusLine: some View {
        switch controller.state {
        case .recording:
            if let startedAt = controller.recordingStartedAt {
                Text("Recording since \(Self.timeFormatter.string(from: startedAt))")
            } else {
                Text("Recording")
            }
        case .paused:
            Text("Paused")
        case .stopped:
            Text("Stopped")
        }
    }

    @ViewBuilder
    private var pauseResumeButton: some View {
        switch controller.state {
        case .recording:
            Button("Pause") { controller.pause() }
        case .paused:
            Button("Resume") { controller.resume() }
        case .stopped:
            // `resume()` only acts on a paused controller, so a stopped one needs `start()`.
            Button("Start") { controller.start() }
        }
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
