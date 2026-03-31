/// Emotion validation utility
/// Validates that emotions match the 7 backend-trained emotions
class EmotionMapper {
  /// List of valid backend emotions (7 total)
  static const List<String> validEmotions = [
    'angry',
    'disgust',
    'fear',
    'happy',
    'neutral',
    'sad',
    'surprise',
  ];

  /// Validate if emotion is a valid backend emotion
  static bool isValidEmotion(String emotion) {
    return validEmotions.contains(emotion.toLowerCase());
  }

  /// Map confidence to intensity
  /// Confidence 0.0-1.0 -> Intensity 30-100
  static int mapConfidenceToIntensity(double confidence) {
    final baseIntensity = (confidence * 100).round();
    return baseIntensity.clamp(30, 100);
  }
}

