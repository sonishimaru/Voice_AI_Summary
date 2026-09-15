import Foundation
import AVFoundation

/// Captures the microphone via `AVAudioEngine`'s input node and delivers
/// 16 kHz mono PCM buffers to `onBuffer`. The OS prompts for microphone
/// access the first time `start()` actually pulls audio (standard
/// `NSMicrophoneUsageDescription` flow) -- no private API involved.
final class MicCapture {
    /// Called with a converted 16 kHz mono PCM buffer whenever new
    /// microphone audio is available. Invoked on whatever thread
    /// `AVAudioEngine` chooses to run the input tap's callback on (a
    /// real-time-ish audio thread) -- keep this cheap; heavier work
    /// (disk I/O) is handed off to `SegmentWriter`'s own serial queue.
    var onBuffer: ((AVAudioPCMBuffer) -> Void)?

    private let engine = AVAudioEngine()
    private var converter: AVAudioConverter?
    private let outputFormat: AVAudioFormat

    init?() {
        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false) else {
            return nil
        }
        self.outputFormat = format
    }

    func start() throws {
        let input = engine.inputNode
        // Ask the input node for its *actual* hardware format rather than
        // assuming one -- required before installing a tap, and it's the
        // format we must feed to the converter as the "from" side.
        let inputFormat = input.inputFormat(forBus: 0)

        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw CaptureError.converterCreationFailed
        }
        self.converter = converter

        // Defensive: in case start() is ever called twice without an
        // intervening stop().
        input.removeTap(onBus: 0)

        input.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] buffer, _ in
            self?.convertAndDeliver(buffer, using: converter)
        }

        engine.prepare()
        try engine.start()
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        converter = nil
    }

    private func convertAndDeliver(_ buffer: AVAudioPCMBuffer, using converter: AVAudioConverter) {
        let ratio = outputFormat.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 32
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else { return }

        var error: NSError?
        // One-shot input block: hand the converter our single source
        // buffer once, then report "no more data" so `convert(to:...)`
        // returns after draining it instead of asking again forever.
        var consumed = false
        converter.convert(to: outBuffer, error: &error) { _, outStatus in
            if consumed {
                outStatus.pointee = .noDataNow
                return nil
            }
            consumed = true
            outStatus.pointee = .haveData
            return buffer
        }

        if let error = error {
            Log.mic.error("Mic audio conversion failed: \(error.localizedDescription, privacy: .public)")
            return
        }
        guard outBuffer.frameLength > 0 else { return }
        onBuffer?(outBuffer)
    }
}
