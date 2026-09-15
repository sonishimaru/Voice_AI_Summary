import Foundation
import AudioToolbox
import AVFoundation

/// Captures ALL system audio output (every process, mixed to mono) using a
/// Core Audio "process tap" plus a private aggregate device -- the
/// technique pioneered by the AudioCap sample project
/// (github.com/insidegui/AudioCap). Delivers 16 kHz mono PCM buffers to
/// `onBuffer`.
///
/// Deliberately NOT ScreenCaptureKit: SCK's system-audio capability re-asks
/// for Screen Recording permission roughly monthly, which is a poor fit for
/// an always-on background recorder. The process-tap route only needs the
/// OS's one-time "system audio recording" prompt, tied to
/// `NSAudioCaptureUsageDescription` -- no private TCC API is used; we just
/// call the public `AudioHardwareCreateProcessTap` API and let the system
/// show its own permission UI on first use.
///
/// NOTE: everything in this file targets the Core Audio "process tap" API
/// introduced in macOS 14.2 (`CATapDescription`, `AudioHardwareCreateProcessTap`,
/// `AudioHardwareCreateAggregateDevice` with a tap entry). The exact symbol
/// names/property selectors below are transcribed from the public
/// AudioCap reference implementation; please double check them against the
/// CoreAudio.framework headers in Xcode 15.3+ on a real Mac before
/// shipping, since this environment cannot compile Swift/Core Audio code.
final class ProcessTapCapture {
    /// Called with a converted 16 kHz mono PCM buffer whenever new system
    /// audio is available. Invoked from the Core Audio IO block registered
    /// on `ioQueue` below -- keep this cheap; heavier work (disk I/O) is
    /// handed off to `SegmentWriter`'s own serial queue.
    var onBuffer: ((AVAudioPCMBuffer) -> Void)?

    private var tapID: AudioObjectID = 0
    private var aggregateDeviceID: AudioObjectID = 0
    private var ioProcID: AudioDeviceIOProcID?

    private var tapFormat: AVAudioFormat?
    private var converter: AVAudioConverter?
    private let outputFormat: AVAudioFormat

    // Core Audio's block-based IO proc is handed this queue; it does its
    // own real-time-safe dispatch under the hood. We still keep our own
    // handler minimal (no allocation beyond what the converter needs).
    private let ioQueue = DispatchQueue(label: "com.voiceaisummary.recorder.systemaudio.io")

    init?() {
        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false) else {
            return nil
        }
        self.outputFormat = format
    }

    func start() throws {
        // 1) Describe a system-wide tap: passing an empty "exclude" list
        //    means no process is excluded, i.e. capture everything the Mac
        //    plays (remote call participants, videos, everything) as one
        //    mono mix. `isMono` is set explicitly per spec.
        let tapDescription = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
        // NOTE: verify `isMono` is a settable property on `CATapDescription`
        // in the real SDK and that, combined with the "stereo" initializer
        // above, it yields a genuine single-channel mix. If the SDK
        // instead exposes a dedicated `monoGlobalTapButExcludeProcesses:`
        // initializer, prefer that instead and drop this line.
        tapDescription.isMono = true
        // Private tap: other processes cannot discover/attach to it.
        // (Obj-C `privateTap` is imported into Swift as `isPrivate`.)
        tapDescription.isPrivate = true
        // Don't mute anything while tapped -- this is a silent bystander,
        // not an exclusive recorder; the user should still hear everything
        // normally.
        tapDescription.muteBehavior = .unmuted
        let tapUUID = UUID()
        tapDescription.uuid = tapUUID

        var newTapID: AudioObjectID = 0
        let tapStatus = AudioHardwareCreateProcessTap(tapDescription, &newTapID)
        guard tapStatus == noErr else { throw CaptureError.tapCreationFailed(tapStatus) }
        tapID = newTapID

        // 2) The aggregate device needs a real physical device to anchor
        //    to (its "sub-device") -- use whatever the user's current
        //    default output device is, so the tap tracks what's actually
        //    audible.
        let outputDeviceID = try Self.readDefaultOutputDeviceID()
        let outputUID = try Self.readDeviceUID(outputDeviceID)

        let aggregateUID = UUID().uuidString
        let aggregateDescription: [String: Any] = [
            kAudioAggregateDeviceNameKey as String: "VoiceAISummary-SystemTap",
            kAudioAggregateDeviceUIDKey as String: aggregateUID,
            kAudioAggregateDeviceMainSubDeviceKey as String: outputUID,
            kAudioAggregateDeviceIsPrivateKey as String: true,
            kAudioAggregateDeviceIsStackedKey as String: false,
            kAudioAggregateDeviceTapAutoStartKey as String: true,
            kAudioAggregateDeviceSubDeviceListKey as String: [
                [kAudioSubDeviceUIDKey as String: outputUID]
            ],
            kAudioAggregateDeviceTapListKey as String: [
                [
                    kAudioSubTapDriftCompensationKey as String: true,
                    kAudioSubTapUIDKey as String: tapUUID.uuidString
                ]
            ]
        ]

        var newAggregateID: AudioObjectID = 0
        let aggStatus = AudioHardwareCreateAggregateDevice(aggregateDescription as CFDictionary, &newAggregateID)
        guard aggStatus == noErr else { throw CaptureError.aggregateDeviceCreationFailed(aggStatus) }
        aggregateDeviceID = newAggregateID

        // 3) Find out what format the tap actually delivers (typically the
        //    hardware's nominal sample rate, e.g. 48 kHz, mono because we
        //    asked for that above), then build a converter down to our
        //    16 kHz mono storage format.
        var asbd = try Self.readTapStreamDescription(newTapID)
        guard let tapFormat = AVAudioFormat(streamDescription: &asbd) else {
            throw CaptureError.formatUnavailable
        }
        self.tapFormat = tapFormat
        guard let converter = AVAudioConverter(from: tapFormat, to: outputFormat) else {
            throw CaptureError.converterCreationFailed
        }
        self.converter = converter

        // 4) Register the IO block on the aggregate device and start it.
        var newProcID: AudioDeviceIOProcID?
        let ioBlock: AudioDeviceIOBlock = { [weak self] _, inInputData, _, _, _ in
            self?.handleIO(inInputData)
        }
        let procStatus = AudioDeviceCreateIOProcIDWithBlock(&newProcID, newAggregateID, ioQueue, ioBlock)
        guard procStatus == noErr, let procID = newProcID else {
            throw CaptureError.ioProcCreationFailed(procStatus)
        }
        ioProcID = procID

        let startStatus = AudioDeviceStart(newAggregateID, procID)
        guard startStatus == noErr else {
            throw CaptureError.ioProcCreationFailed(startStatus)
        }
    }

    func stop() {
        if let procID = ioProcID {
            AudioDeviceStop(aggregateDeviceID, procID)
            AudioDeviceDestroyIOProcID(aggregateDeviceID, procID)
            ioProcID = nil
        }
        if aggregateDeviceID != 0 {
            AudioHardwareDestroyAggregateDevice(aggregateDeviceID)
            aggregateDeviceID = 0
        }
        if tapID != 0 {
            // NOTE: verify this exact destroy-function name in the real
            // SDK; it's the documented symmetric counterpart to
            // `AudioHardwareCreateProcessTap` following Core Audio's
            // Create/Destroy naming convention used elsewhere in this file.
            AudioHardwareDestroyProcessTap(tapID)
            tapID = 0
        }
        converter = nil
        tapFormat = nil
    }

    // MARK: - IO callback

    /// Runs on the queue handed to `AudioDeviceCreateIOProcIDWithBlock`.
    /// Wraps the raw `AudioBufferList` in an `AVAudioPCMBuffer` with no
    /// copy, converts it down to 16 kHz mono, and forwards it.
    private func handleIO(_ inputData: UnsafePointer<AudioBufferList>) {
        guard let tapFormat = tapFormat, let converter = converter else { return }
        guard let inBuffer = AVAudioPCMBuffer(pcmFormat: tapFormat, bufferListNoCopy: inputData, deallocator: nil) else { return }
        guard inBuffer.frameLength > 0 else { return }

        let ratio = outputFormat.sampleRate / tapFormat.sampleRate
        let capacity = AVAudioFrameCount(Double(inBuffer.frameLength) * ratio) + 32
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else { return }

        var error: NSError?
        var consumed = false
        converter.convert(to: outBuffer, error: &error) { _, outStatus in
            if consumed {
                outStatus.pointee = .noDataNow
                return nil
            }
            consumed = true
            outStatus.pointee = .haveData
            return inBuffer
        }

        if let error = error {
            Log.systemAudio.error("System audio conversion failed: \(error.localizedDescription, privacy: .public)")
            return
        }
        guard outBuffer.frameLength > 0 else { return }
        onBuffer?(outBuffer)
    }

    // MARK: - Core Audio property readers

    static func readDefaultOutputDeviceID() throws -> AudioObjectID {
        var deviceID = AudioObjectID(0)
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        let status = AudioObjectGetPropertyData(kAudioObjectSystemObject, &address, 0, nil, &size, &deviceID)
        guard status == noErr else { throw CaptureError.propertyReadFailed(status) }
        return deviceID
    }

    static func readDeviceUID(_ deviceID: AudioObjectID) throws -> String {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var uid: CFString = "" as CFString
        var size = UInt32(MemoryLayout<CFString>.size)
        let status = withUnsafeMutablePointer(to: &uid) { ptr -> OSStatus in
            AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, ptr)
        }
        guard status == noErr else { throw CaptureError.propertyReadFailed(status) }
        return uid as String
    }

    /// NOTE: `kAudioTapPropertyFormat` is transcribed from the AudioCap
    /// reference implementation's approach to reading a process tap's
    /// stream format; please confirm the exact selector name against
    /// CoreAudio.framework's tapping headers in a real Xcode 15.3+ install.
    static func readTapStreamDescription(_ tapID: AudioObjectID) throws -> AudioStreamBasicDescription {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioTapPropertyFormat,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var asbd = AudioStreamBasicDescription()
        var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        let status = AudioObjectGetPropertyData(tapID, &address, 0, nil, &size, &asbd)
        guard status == noErr else { throw CaptureError.propertyReadFailed(status) }
        return asbd
    }
}
