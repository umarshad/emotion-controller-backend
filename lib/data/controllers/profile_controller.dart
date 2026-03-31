import 'package:get/get.dart';
import '../models/user_profile.dart';
import '../models/mood_streak.dart';
import '../models/emotion_stats.dart';
import '../models/emotion_model.dart';
import 'emotion_controller.dart';

/// Controller for managing user profile and emotional statistics
class ProfileController extends GetxController {
  // User profile
  final Rx<UserProfile> userProfile = UserProfile.defaultUser().obs;

  // Emotional statistics
  final Rx<EmotionStats> emotionStats = EmotionStats.empty().obs;

  // Mood streak
  final Rx<MoodStreak> moodStreak = MoodStreak.empty().obs;

  @override
  void onInit() {
    super.onInit();
    final emotionController = Get.find<EmotionController>();

    // Initial load
    _loadProfileData();

    // Listen for changes in emotions list to update stats automatically
    ever(emotionController.emotions, (_) {
      _calculateEmotionStats(emotionController.emotions);
      _calculateMoodStreak(emotionController.emotions);
    });
  }

  /// Load profile data and calculate statistics
  void _loadProfileData() {
    final emotionController = Get.find<EmotionController>();
    _calculateEmotionStats(emotionController.emotions);
    _calculateMoodStreak(emotionController.emotions);
  }

  /// Calculate emotional statistics from emotions
  void _calculateEmotionStats(List<EmotionModel> emotions) {
    if (emotions.isEmpty) {
      emotionStats.value = EmotionStats.empty();
      return;
    }

    final now = DateTime.now();
    final weekStart = now.subtract(Duration(days: now.weekday - 1));
    final monthStart = DateTime(now.year, now.month, 1);

    final emotionCounts = <String, int>{};
    int totalIntensity = 0;
    int thisWeekCount = 0;
    int thisMonthCount = 0;

    for (final emotion in emotions) {
      // Count emotions
      emotionCounts[emotion.type] = (emotionCounts[emotion.type] ?? 0) + 1;
      totalIntensity += emotion.intensity;

      // Count this week
      if (emotion.timestamp.isAfter(
        weekStart.subtract(const Duration(days: 1)),
      )) {
        thisWeekCount++;
      }

      // Count this month
      if (emotion.timestamp.isAfter(
        monthStart.subtract(const Duration(days: 1)),
      )) {
        thisMonthCount++;
      }
    }

    // Find most frequent emotion
    String? mostFrequent;
    int mostFrequentCount = 0;
    emotionCounts.forEach((emotion, count) {
      if (count > mostFrequentCount) {
        mostFrequent = emotion;
        mostFrequentCount = count;
      }
    });

    emotionStats.value = EmotionStats(
      totalEmotions: emotions.length,
      mostFrequentEmotion: mostFrequent,
      mostFrequentCount: mostFrequentCount,
      emotionCounts: emotionCounts,
      averageIntensity: totalIntensity / emotions.length,
      thisWeekCount: thisWeekCount,
      thisMonthCount: thisMonthCount,
    );
  }

  /// Calculate mood streak
  void _calculateMoodStreak(List<EmotionModel> emotions) {
    if (emotions.isEmpty) {
      moodStreak.value = MoodStreak.empty();
      return;
    }

    // Sort by date descending
    final sorted = List<EmotionModel>.from(emotions)
      ..sort((a, b) => b.timestamp.compareTo(a.timestamp));

    int currentStreak = 0;
    int longestStreak = 0;
    int tempStreak = 0;
    DateTime? lastDate;
    String? currentStreakEmotion;

    // Calculate streaks
    for (int i = 0; i < sorted.length; i++) {
      final emotion = sorted[i];
      final emotionDate = DateTime(
        emotion.timestamp.year,
        emotion.timestamp.month,
        emotion.timestamp.day,
      );

      if (lastDate == null) {
        // First emotion
        currentStreak = 1;
        tempStreak = 1;
        lastDate = emotionDate;
        currentStreakEmotion = emotion.type;
      } else {
        final daysDiff = lastDate.difference(emotionDate).inDays;

        if (daysDiff == 1) {
          // Consecutive day
          if (i == 0) {
            currentStreak++;
          }
          tempStreak++;
        } else if (daysDiff == 0) {
          // Same day - continue streak
          if (i == 0) {
            currentStreakEmotion = emotion.type;
          }
        } else {
          // Streak broken
          if (tempStreak > longestStreak) {
            longestStreak = tempStreak;
          }
          tempStreak = 1;
        }

        lastDate = emotionDate;
      }
    }

    if (tempStreak > longestStreak) {
      longestStreak = tempStreak;
    }

    moodStreak.value = MoodStreak(
      currentStreak: currentStreak,
      longestStreak: longestStreak,
      lastCheckInDate: sorted.isNotEmpty ? sorted.first.timestamp : null,
      currentStreakEmotion: currentStreakEmotion,
    );
  }

  /// Update user profile
  void updateProfile(UserProfile profile) {
    userProfile.value = profile;
  }

  /// Refresh statistics
  void refreshStats() {
    final emotionController = Get.find<EmotionController>();
    _calculateEmotionStats(emotionController.emotions);
    _calculateMoodStreak(emotionController.emotions);
  }
}
