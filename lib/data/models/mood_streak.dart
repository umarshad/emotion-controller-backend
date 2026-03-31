/// Mood streak model
class MoodStreak {
  final int currentStreak;
  final int longestStreak;
  final DateTime? lastCheckInDate;
  final String? currentStreakEmotion;

  MoodStreak({
    required this.currentStreak,
    required this.longestStreak,
    this.lastCheckInDate,
    this.currentStreakEmotion,
  });

  factory MoodStreak.empty() {
    return MoodStreak(
      currentStreak: 0,
      longestStreak: 0,
    );
  }
}
