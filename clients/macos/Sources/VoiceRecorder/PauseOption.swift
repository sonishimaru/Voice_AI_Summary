import Foundation

/// The pause durations offered by both menus (the Dock menu and the
/// optional menu bar one), so the two can never drift apart.
///
/// `Int` raw values are what the Dock menu puts in `NSMenuItem.tag` to get
/// from a clicked item back to the option, so they must stay stable.
enum PauseOption: Int, CaseIterable {
    case thirtyMinutes = 0
    case oneHour = 1
    case untilTomorrow = 2
    case untilResumed = 3

    var title: String {
        switch self {
        case .thirtyMinutes: return "30 分"
        case .oneHour: return "1 時間"
        case .untilTomorrow: return "今日中"
        case .untilResumed: return "再開するまで"
        }
    }

    /// When recording should resume, or `nil` for "not until the user says
    /// so". `.untilTomorrow` means the next local midnight -- so choosing it
    /// in the evening resumes in a few hours, not 24 of them.
    func resumeDate(from now: Date = Date()) -> Date? {
        switch self {
        case .thirtyMinutes:
            return now.addingTimeInterval(30 * 60)
        case .oneHour:
            return now.addingTimeInterval(60 * 60)
        case .untilTomorrow:
            return Calendar.current.nextDate(
                after: now,
                matching: DateComponents(hour: 0, minute: 0),
                matchingPolicy: .nextTime
            ) ?? now.addingTimeInterval(24 * 60 * 60)
        case .untilResumed:
            return nil
        }
    }
}
