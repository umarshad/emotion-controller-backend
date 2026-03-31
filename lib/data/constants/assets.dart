/// Asset paths and emoji mappings for emotions
/// Matches backend trained emotions: angry, disgust, fear, happy, neutral, sad, surprise
class AppAssets {
  // Emotion Emojis (7 backend emotions)
  static const String emojiAngry = '😠';
  static const String emojiDisgust = '🤢';
  static const String emojiFear = '😨';
  static const String emojiHappy = '😊';
  static const String emojiNeutral = '😐';
  static const String emojiSad = '😢';
  static const String emojiSurprise = '😲';

  // Navigation Icons (using emojis for simplicity)
  static const String iconHome = '🏠';
  static const String iconChat = '💬';
  static const String iconCamera = '📷';
  static const String iconHistory = '📊';
  static const String iconProfile = '⚙️';

  /// Get emoji for emotion type
  /// Returns emoji for the 7 backend-trained emotions
  static String getEmotionEmoji(String emotionType) {
    switch (emotionType.toLowerCase()) {
      case 'angry':
        return emojiAngry;
      case 'disgust':
        return emojiDisgust;
      case 'fear':
        return emojiFear;
      case 'happy':
        return emojiHappy;
      case 'neutral':
        return emojiNeutral;
      case 'sad':
        return emojiSad;
      case 'surprise':
        return emojiSurprise;
      default:
        return emojiNeutral; // Default to neutral
    }
  }

  /// Get display name for emotion type
  /// Returns display name for the 7 backend-trained emotions
  static String getEmotionName(String emotionType) {
    switch (emotionType.toLowerCase()) {
      case 'angry':
        return 'Angry';
      case 'disgust':
        return 'Disgust';
      case 'fear':
        return 'Fear';
      case 'happy':
        return 'Happy';
      case 'neutral':
        return 'Neutral';
      case 'sad':
        return 'Sad';
      case 'surprise':
        return 'Surprise';
      default:
        return 'Unknown';
    }
  }

  /// List of all supported emotions
  /// Matches backend trained emotions exactly (7 emotions)
  static const List<String> allEmotions = [
    'angry',    // Backend label: 0
    'disgust',  // Backend label: 1
    'fear',     // Backend label: 2
    'happy',    // Backend label: 3
    'neutral',  // Backend label: 4
    'sad',      // Backend label: 5
    'surprise', // Backend label: 6
  ];
}