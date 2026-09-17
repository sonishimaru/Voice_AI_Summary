import AppKit

/// Draws the recorder's state onto the Dock icon.
///
/// This app ships no icon asset, so without this the Dock would show the
/// generic blank-document icon. Rendering the same SF Symbol the menu bar
/// uses means the Dock tile is both the app icon and the state indicator:
/// a red waveform while recording, an orange pause bar while paused, a
/// grey stop square while stopped -- readable at a glance from across the
/// screen, which a menu bar item squeezed next to the notch is not.
enum DockIcon {
    /// Point size the symbol is rendered at. The Dock scales the tile to
    /// whatever size the user's Dock is set to, so this only needs to be
    /// large enough that scaling down stays sharp.
    private static let pointSize: CGFloat = 512

    static func symbolName(for state: RecordingController.State) -> String {
        switch state {
        case .recording: return "waveform.circle.fill"
        case .paused: return "pause.circle.fill"
        case .stopped: return "stop.circle"
        }
    }

    static func color(for state: RecordingController.State) -> NSColor {
        switch state {
        case .recording: return .systemRed
        case .paused: return .systemOrange
        case .stopped: return .systemGray
        }
    }

    /// Replaces the Dock tile's image with the symbol for `state`.
    ///
    /// A failure to build the image is not worth surfacing: the Dock just
    /// keeps whatever it was showing, which is strictly better than
    /// clearing it to nothing.
    @MainActor
    static func update(for state: RecordingController.State) {
        let configuration = NSImage.SymbolConfiguration(pointSize: pointSize, weight: .regular)
            .applying(NSImage.SymbolConfiguration(paletteColors: [color(for: state)]))
        guard
            let symbol = NSImage(systemSymbolName: symbolName(for: state), accessibilityDescription: "録音の状態"),
            let image = symbol.withSymbolConfiguration(configuration)
        else {
            Log.app.error("Could not build a Dock icon for state \(String(describing: state), privacy: .public).")
            return
        }
        NSApplication.shared.applicationIconImage = image
    }
}
