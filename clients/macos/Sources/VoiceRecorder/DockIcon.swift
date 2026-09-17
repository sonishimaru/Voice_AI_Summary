import AppKit

/// Draws the recorder's state as the Dock icon.
///
/// This app ships no icon asset, so without this the Dock shows the generic
/// blank-document icon. Rather than drop a bare SF Symbol on the tile --
/// which reads as a stray glyph rather than an app -- the icon is drawn as a
/// proper rounded-square app icon whose fill colour *is* the state:
///
///   recording  red     waveform
///   paused     amber   pause bars
///   stopped    grey    crossed-out mic
///
/// Colour carries across the room, the glyph disambiguates up close, and
/// neither alone has to do the whole job -- which also keeps it readable for
/// anyone who cannot tell the red from the amber.
enum DockIcon {
    /// Rendered once per state and reused; redrawing a 1024pt tile on every
    /// pause/resume would be wasteful, and there are only three of them.
    /// `@MainActor` because it is mutable shared state -- only `update(for:)`
    /// touches it, and that runs on the main actor.
    @MainActor private static var cache: [String: NSImage] = [:]

    /// Tile size to render at. The Dock scales this down to whatever size the
    /// user's Dock is set to, so it only has to be big enough to stay sharp.
    private static let tileSize: CGFloat = 1024

    /// Transparent margin around the art, as a fraction of the tile. macOS app
    /// icons do not run to the edge of their tile; matching that keeps this one
    /// from looking oversized next to its neighbours in the Dock.
    private static let insetRatio: CGFloat = 0.09

    /// Corner radius as a fraction of the art's width -- approximates the
    /// macOS app-icon squircle closely enough at Dock sizes.
    private static let cornerRatio: CGFloat = 0.225

    /// Glyph height as a fraction of the art's width.
    private static let glyphRatio: CGFloat = 0.46

    static func symbolName(for state: RecordingController.State) -> String {
        switch state {
        case .recording: return "waveform"
        case .paused: return "pause.fill"
        // Not `stop.fill`: what matters when the recorder is off is that
        // nothing is being listened to, and a struck-through mic says that
        // outright.
        case .stopped: return "mic.slash.fill"
        }
    }

    /// Explicit sRGB rather than `NSColor.systemRed` and friends: the gradient
    /// is derived from this colour, and a system colour that shifts with
    /// appearance or accessibility settings would take the derived shades with
    /// it, in ways that are not visible from here.
    static func color(for state: RecordingController.State) -> NSColor {
        switch state {
        case .recording: return NSColor(srgbRed: 0.85, green: 0.21, blue: 0.22, alpha: 1)
        case .paused: return NSColor(srgbRed: 0.95, green: 0.62, blue: 0.11, alpha: 1)
        case .stopped: return NSColor(srgbRed: 0.42, green: 0.45, blue: 0.50, alpha: 1)
        }
    }

    /// The finished tile for `state`. A pure function: the drawing handler
    /// below can be invoked lazily, on whatever thread ends up rendering the
    /// image, so it captures only locals and never reaches for shared state.
    static func image(for state: RecordingController.State) -> NSImage? {
        let base = color(for: state)
        let top = base.blended(withFraction: 0.20, of: .white) ?? base
        let bottom = base.blended(withFraction: 0.16, of: .black) ?? base

        guard let glyph = glyphImage(for: state) else { return nil }

        let size = NSSize(width: tileSize, height: tileSize)
        let image = NSImage(size: size, flipped: false) { rect in
            let inset = rect.width * insetRatio
            let art = rect.insetBy(dx: inset, dy: inset)
            let radius = art.width * cornerRatio
            let path = NSBezierPath(roundedRect: art, xRadius: radius, yRadius: radius)

            // `angle: 270` points the gradient downwards, so the starting
            // colour lands at the top.
            NSGradient(starting: top, ending: bottom)?.draw(in: path, angle: 270)

            // A faint inner edge, so the tile still has a defined shape
            // against a light wallpaper.
            NSColor.white.withAlphaComponent(0.18).setStroke()
            path.lineWidth = rect.width * 0.006
            path.stroke()

            let origin = NSPoint(
                x: rect.midX - glyph.size.width / 2,
                y: rect.midY - glyph.size.height / 2
            )
            glyph.draw(at: origin, from: .zero, operation: .sourceOver, fraction: 1)
            return true
        }
        return image
    }

    private static func glyphImage(for state: RecordingController.State) -> NSImage? {
        let configuration = NSImage.SymbolConfiguration(
            pointSize: tileSize * (1 - 2 * insetRatio) * glyphRatio,
            weight: .semibold
        ).applying(NSImage.SymbolConfiguration(paletteColors: [.white]))
        return NSImage(systemSymbolName: symbolName(for: state), accessibilityDescription: "録音の状態")?
            .withSymbolConfiguration(configuration)
    }

    /// Replaces the Dock tile's image with the icon for `state`.
    ///
    /// A failure to build the image is not worth surfacing: the Dock keeps
    /// whatever it was showing, which beats clearing it to nothing.
    @MainActor
    static func update(for state: RecordingController.State) {
        let key = symbolName(for: state)
        // Named `tile`, not `image`, so the call below unambiguously refers to
        // the static method rather than the binding being introduced here.
        guard let tile = cache[key] ?? image(for: state) else {
            Log.app.error("Could not build a Dock icon for state \(String(describing: state), privacy: .public).")
            return
        }
        cache[key] = tile
        NSApplication.shared.applicationIconImage = tile
    }
}
