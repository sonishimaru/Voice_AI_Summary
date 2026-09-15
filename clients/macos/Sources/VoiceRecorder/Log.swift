import OSLog

/// Central place for the app's `os.Logger` instances so the subsystem and
/// category strings are defined exactly once and stay consistent.
enum Log {
    private static let subsystem = "com.voiceaisummary.recorder"

    static let app = Logger(subsystem: subsystem, category: "app")
    static let mic = Logger(subsystem: subsystem, category: "mic")
    static let systemAudio = Logger(subsystem: subsystem, category: "system-audio")
    static let writer = Logger(subsystem: subsystem, category: "segment-writer")
    static let controller = Logger(subsystem: subsystem, category: "controller")
}

/// Errors thrown by the two capture classes (`MicCapture`, `ProcessTapCapture`).
/// Kept in one place since both classes throw the same small set of cases.
enum CaptureError: Error {
    /// `AVAudioConverter(from:to:)` returned nil (formats deemed incompatible).
    case converterCreationFailed
    /// `AudioHardwareCreateProcessTap` failed; associated value is the `OSStatus`.
    case tapCreationFailed(OSStatus)
    /// `AudioHardwareCreateAggregateDevice` failed; associated value is the `OSStatus`.
    case aggregateDeviceCreationFailed(OSStatus)
    /// `AudioDeviceCreateIOProcIDWithBlock` or `AudioDeviceStart` failed.
    case ioProcCreationFailed(OSStatus)
    /// Could not read a required Core Audio property (device UID, tap format, ...).
    case propertyReadFailed(OSStatus)
    /// A required audio format could not be constructed.
    case formatUnavailable
}
