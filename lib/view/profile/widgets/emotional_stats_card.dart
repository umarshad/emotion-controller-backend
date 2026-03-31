import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import '../../../data/constants/colors.dart';
import '../../../data/constants/assets.dart';
import '../../../data/controllers/profile_controller.dart';
import '../../../widgets/custom_text.dart';
import '../../../widgets/statistics_card.dart';

/// Emotional statistics card for profile
class EmotionalStatsCard extends StatelessWidget {
  const EmotionalStatsCard({super.key});

  @override
  Widget build(BuildContext context) {
    final profileController = Get.find<ProfileController>();

    return Obx(() {
      final stats = profileController.emotionStats.value;

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
                Icon(Icons.analytics, color: AppColors.primary, size: 24.sp),
                SizedBox(width: 8.w),
                CustomText(
                  'Emotional Statistics',
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ],
            ),
            SizedBox(height: 20.h),
            Row(
              children: [
                Expanded(
                  child: StatisticsCard(
                    title: 'Total',
                    value: '${stats.totalEmotions}',
                    subtitle: 'emotions',
                    icon: Icons.favorite,
                    iconColor: AppColors.primary,
                  ),
                ),
                SizedBox(width: 12.w),
                Expanded(
                  child: StatisticsCard(
                    title: 'This Week',
                    value: '${stats.thisWeekCount}',
                    subtitle: 'detections',
                    icon: Icons.calendar_today,
                    iconColor: AppColors.secondary,
                  ),
                ),
              ],
            ),
            SizedBox(height: 12.h),
            if (stats.mostFrequentEmotion != null) ...[
              Container(
                padding: EdgeInsets.all(16.w),
                decoration: BoxDecoration(
                  color: getEmotionColor(
                    stats.mostFrequentEmotion!,
                  ).withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(12.r),
                ),
                child: Row(
                  children: [
                    Text(
                      AppAssets.getEmotionEmoji(stats.mostFrequentEmotion!),
                      style: TextStyle(fontSize: 32.sp),
                    ),
                    SizedBox(width: 12.w),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          CustomText(
                            'Most Frequent',
                            fontSize: 12,
                            color: AppColors.textSecondary,
                          ),
                          SizedBox(height: 4.h),
                          CustomText(
                            AppAssets.getEmotionName(
                              stats.mostFrequentEmotion!,
                            ),
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: getEmotionColor(stats.mostFrequentEmotion!),
                          ),
                          SizedBox(height: 2.h),
                          CustomText(
                            '${stats.mostFrequentCount} times',
                            fontSize: 12,
                            color: AppColors.textLight,
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ],
        ),
      );
    });
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
