/// Emotional statistics model
class EmotionStats {
  final int totalEmotions;
  final String? mostFrequentEmotion;
  final int mostFrequentCount;
  final Map<String, int> emotionCounts;
  final double averageIntensity;
  final int thisWeekCount;
  final int thisMonthCount;

  EmotionStats({
    required this.totalEmotions,
    this.mostFrequentEmotion,
    this.mostFrequentCount = 0,
    required this.emotionCounts,
    required this.averageIntensity,
    required this.thisWeekCount,
    required this.thisMonthCount,
  });

  factory EmotionStats.empty() {
    return EmotionStats(
      totalEmotions: 0,
      emotionCounts: {},
      averageIntensity: 0.0,
      thisWeekCount: 0,
      thisMonthCount: 0,
    );
  }
}
