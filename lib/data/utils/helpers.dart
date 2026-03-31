import 'package:intl/intl.dart';
import '../models/emotion_model.dart';

/// Helper functions for date formatting and utilities
class Helpers {
  /// Format date for display
  static String formatDate(DateTime date) {
    final now = DateTime.now();
    final difference = now.difference(date);

    if (difference.inDays == 0) {
      if (difference.inHours == 0) {
        if (difference.inMinutes == 0) {
          return 'Just now';
        }
        return '${difference.inMinutes} minutes ago';
      }
      return '${difference.inHours} hours ago';
    } else if (difference.inDays == 1) {
      return 'Yesterday';
    } else if (difference.inDays < 7) {
      return '${difference.inDays} days ago';
    } else {
      return DateFormat('MMM dd, yyyy').format(date);
    }
  }

  /// Format time for display
  static String formatTime(DateTime date) {
    return DateFormat('hh:mm a').format(date);
  }

  /// Format full date and time
  static String formatDateTime(DateTime date) {
    return DateFormat('MMM dd, yyyy • hh:mm a').format(date);
  }

  /// Get intensity level text
  static String getIntensityLevel(int intensity) {
    if (intensity < 25) return 'Low';
    if (intensity < 50) return 'Moderate';
    if (intensity < 75) return 'High';
    return 'Very High';
  }

  /// Get intensity color based on value
  static String getIntensityColor(int intensity) {
    if (intensity < 25) return 'low';
    if (intensity < 50) return 'moderate';
    if (intensity < 75) return 'high';
    return 'veryHigh';
  }

  /// Group emotions by date
  static Map<String, List<EmotionModel>> groupEmotionsByDate(
      List<EmotionModel> emotions) {
    final Map<String, List<EmotionModel>> grouped = {};

    for (final emotion in emotions) {
      final dateKey = DateFormat('yyyy-MM-dd').format(emotion.timestamp);
      if (!grouped.containsKey(dateKey)) {
        grouped[dateKey] = [];
      }
      grouped[dateKey]!.add(emotion);
    }

    return grouped;
  }

  /// Check if date is today
  static bool isToday(DateTime date) {
    final now = DateTime.now();
    return date.year == now.year &&
        date.month == now.month &&
        date.day == now.day;
  }

  /// Check if date is yesterday
  static bool isYesterday(DateTime date) {
    final yesterday = DateTime.now().subtract(const Duration(days: 1));
    return date.year == yesterday.year &&
        date.month == yesterday.month &&
        date.day == yesterday.day;
  }

  /// Check if date is this week
  static bool isThisWeek(DateTime date) {
    final now = DateTime.now();
    final weekStart = now.subtract(Duration(days: now.weekday - 1));
    return date.isAfter(weekStart.subtract(const Duration(days: 1)));
  }
}