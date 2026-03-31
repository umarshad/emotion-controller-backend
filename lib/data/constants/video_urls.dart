import 'package:youtube_player_flutter/youtube_player_flutter.dart';

/// Video URL mappings for emotions
/// Maps each emotion type to appropriate YouTube video URL
class EmotionVideoUrls {
  // Video URLs - confirmed embeddable
  static const String angerVideo = 'https://youtube.com/shorts/XT5HNfeacCA';
  static const String anxietyVideo =
      'https://www.youtube.com/watch?v=30VMIEmA114';
  static const String breathingVideo =
      'https://www.youtube.com/watch?v=un47AZivijQ';
  static const String video1 = 'https://youtu.be/4RHAznBTU1U';
  static const String video2 = 'https://youtu.be/DbpTohPUhMw';

  /// Get video URL for emotion type
  /// Maps 7 backend emotions to appropriate videos
  static String getVideoUrl(String emotionType) {
    switch (emotionType.toLowerCase()) {
      case 'angry':
        return angerVideo;
      case 'disgust':
      case 'fear':
      case 'sad':
        return video1;
      case 'happy':
      case 'surprise':
        return video2;
      case 'neutral':
        return breathingVideo;
      default:
        return breathingVideo; // Default to breathing video
    }
  }

  /// Music videos for stress relief (all in-app embeddable)
  static const List<Map<String, String>> calmingMusicVideos = [
    {
      'title': 'Peaceful Meditation Music',
      'url': 'https://youtu.be/DzYp5uqixz0',
    },
    {'title': 'Calming Ambient Music', 'url': 'https://youtu.be/htnWMNfK1nc'},
    {
      'title': 'Meditation Music Free',
      'url': 'https://www.youtube.com/watch?v=fAsru_unBUE',
    },
    {
      'title': 'Anger Release Free Music',
      'url': 'https://www.youtube.com/watch?v=QbE10b9Jh9A',
    },
  ];

  /// Quranic recitation videos
  static const List<Map<String, String>> quranVideos = [
    {
      'title': 'Surah Al-Mulk Recitation',
      'url': 'https://youtu.be/Dai9lZ4Sne0',
    },
    {
      'title': 'Beautiful Quran Recitation',
      'url': 'https://youtu.be/dkTd3XQFr8g',
    },
  ];

  /// Funny / feel-good videos
  static const List<Map<String, String>> funnyVideos = [
    {'title': 'Funny Clean Comedy', 'url': 'https://youtu.be/d8e4qjZwAE8'},
    {'title': 'Hilarious Moments', 'url': 'https://youtu.be/7FDAJ8L3lig'},
  ];

  /// Calming & grounding technique videos
  static const List<Map<String, String>> calmingVideos = [
    {
      'title': 'Grounding Technique for Anxiety',
      'url': 'https://youtu.be/Z7C0v4GfUUI',
    },
  ];

  /// Fitness & workout videos
  static const List<Map<String, String>> fitnessVideos = [
    {'title': 'Full Body Workout', 'url': 'https://youtu.be/ibywaQB3L7o'},
    {
      'title': 'Fitness Training Session',
      'url': 'https://youtu.be/cmple9fw65w',
    },
  ];

  /// Returns two embeddable suggested videos for the Reflection screen.
  /// All URLs are the same confirmed-embeddable set, with titles suited to
  /// the reflection / journaling context.
  static List<Map<String, String>> getReflectionVideos(String emotionType) {
    switch (emotionType.toLowerCase()) {
      case 'happy':
        return [
          {'title': 'Boost Your Happiness', 'url': video2},
          {'title': 'Positive Mindset Practice', 'url': anxietyVideo},
        ];
      case 'sad':
        return [
          {'title': 'Healing Through Emotions', 'url': video1},
          {'title': 'Calming Breathing Exercise', 'url': breathingVideo},
        ];
      case 'angry':
        return [
          {'title': '5 Steps to Manage Anger', 'url': angerVideo},
          {'title': 'Releasing Negative Energy', 'url': video1},
        ];
      case 'fear':
        return [
          {'title': 'Facing Your Fears', 'url': video1},
          {'title': 'Breathe Through Anxiety', 'url': breathingVideo},
        ];
      case 'surprise':
        return [
          {'title': 'Embrace the Unexpected', 'url': video2},
          {'title': 'Stay Open-Minded', 'url': anxietyVideo},
        ];
      case 'disgust':
        return [
          {'title': 'Reframing Difficult Emotions', 'url': video1},
          {'title': 'Grounding Yourself', 'url': breathingVideo},
        ];
      case 'neutral':
      default:
        return [
          {'title': 'Mindful Breathing', 'url': breathingVideo},
          {'title': 'Finding Balance', 'url': anxietyVideo},
        ];
    }
  }

  /// Extract video ID from YouTube URL
  static String extractVideoId(String url) {
    return YoutubePlayer.convertUrlToId(url) ?? '';
  }
}
