import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import '../../../data/constants/colors.dart';
import '../../../data/constants/assets.dart';
import '../../../data/controllers/emotion_controller.dart';
import '../../../widgets/custom_text.dart';

/// Emotional insights card with trends and statistics
class EmotionalInsightsCard extends StatelessWidget {
  const EmotionalInsightsCard({super.key});

  @override
  Widget build(BuildContext context) {
    final emotionController = Get.find<EmotionController>();

    return Obx(() {
      final stats = emotionController.getEmotionStats();
      final recentEmotions = emotionController.getRecentEmotions(count: 7);

      if (stats.isEmpty) {
        return const SizedBox.shrink();
      }

      // Find most common emotion this week
      final weekEmotions = recentEmotions
          .where((e) => DateTime.now().difference(e.timestamp).inDays < 7)
          .toList();
      final weekStats = <String, int>{};
      for (final emotion in weekEmotions) {
        weekStats[emotion.type] = (weekStats[emotion.type] ?? 0) + 1;
      }

      String? mostCommonThisWeek;
      int mostCommonCount = 0;
      weekStats.forEach((emotion, count) {
        if (count > mostCommonCount) {
          mostCommonThisWeek = emotion;
          mostCommonCount = count;
        }
      });

      return Container(
        margin: EdgeInsets.symmetric(horizontal: 16.w),
        padding: EdgeInsets.all(20.w),
        decoration: BoxDecoration(
          color: AppColors.cardBackground,
          borderRadius: BorderRadius.circular(16.r),
          border: Border.all(color: AppColors.border),
          boxShadow: [
            BoxShadow(
              color: AppColors.shadow,
              blurRadius: 8,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.insights, color: AppColors.primary, size: 24.sp),
                SizedBox(width: 8.w),
                CustomText(
                  'Emotional Insights',
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ],
            ),
            SizedBox(height: 20.h),
            if (mostCommonThisWeek != null) ...[
              _buildInsightRow(
                'Most Common This Week',
                AppAssets.getEmotionName(mostCommonThisWeek!),
                AppAssets.getEmotionEmoji(mostCommonThisWeek!),
                getEmotionColor(mostCommonThisWeek!),
              ),
              SizedBox(height: 16.h),
            ],
            _buildInsightRow(
              'Total Emotions Detected',
              '${stats.values.reduce((a, b) => a + b)}',
              '📊',
              AppColors.secondary,
            ),
            SizedBox(height: 16.h),
            _buildInsightRow(
              'Emotion Types Tracked',
              '${stats.length} different emotions',
              '🎭',
              AppColors.accent,
            ),
          ],
        ),
      );
    });
  }

  Widget _buildInsightRow(
    String label,
    String value,
    String emoji,
    Color color,
  ) {
    return Row(
      children: [
        Container(
          width: 40.w,
          height: 40.h,
          decoration: BoxDecoration(
            color: color.withValues(alpha: 0.1),
            borderRadius: BorderRadius.circular(10.r),
          ),
          child: Center(
            child: Text(emoji, style: TextStyle(fontSize: 20.sp)),
          ),
        ),
        SizedBox(width: 12.w),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              CustomText(label, fontSize: 12, color: AppColors.textSecondary),
              SizedBox(height: 2.h),
              CustomText(
                value,
                fontSize: 16,
                fontWeight: FontWeight.w600,
                color: color,
              ),
            ],
          ),
        ),
      ],
    );
  }
}

/// Get emotion color helper
/// Uses the 7 backend emotions
Color getEmotionColor(String emotionType) {
  switch (emotionType.toLowerCase()) {
    case 'angry':
      return AppColors.emotionAngry;
    case 'disgust':
      return AppColors.emotionDisgust;
    case 'fear':
      return AppColors.emotionFear;
    case 'happy':
      return AppColors.emotionHappy;
    case 'neutral':
      return AppColors.emotionNeutral;
    case 'sad':
      return AppColors.emotionSad;
    case 'surprise':
      return AppColors.emotionSurprise;
    default:
      return AppColors.primary;
  }
}
