import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import '../data/constants/colors.dart';
import '../data/constants/assets.dart';
import '../data/models/mood_streak.dart';
import 'custom_text.dart';

/// Mood streak visualization widget
class MoodStreakWidget extends StatelessWidget {
  final MoodStreak streak;

  const MoodStreakWidget({super.key, required this.streak});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.all(20.w),
      decoration: BoxDecoration(
        color: Theme.of(context).cardColor,
        borderRadius: BorderRadius.circular(16.r),
        border: Border.all(color: Theme.of(context).dividerColor),
        boxShadow: [
          BoxShadow(
            color: Theme.of(context).shadowColor,
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
              Icon(
                Icons.local_fire_department,
                color: AppColors.emotionHappy,
                size: 24.sp,
              ),
              SizedBox(width: 8.w),
              CustomText(
                'Mood Streak',
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ],
          ),
          SizedBox(height: 16.h),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              Expanded(
                child: _buildStreakItem(
                  context,
                  'Current',
                  streak.currentStreak.toString(),
                  AppColors.primary,
                ),
              ),
              Container(
                width: 1.w,
                height: 40.h,
                color: Theme.of(context).dividerColor,
              ),
              Expanded(
                child: _buildStreakItem(
                  context,
                  'Longest',
                  streak.longestStreak.toString(),
                  AppColors.secondary,
                ),
              ),
            ],
          ),
          if (streak.currentStreakEmotion != null) ...[
            SizedBox(height: 12.h),
            Container(
              padding: EdgeInsets.all(12.w),
              decoration: BoxDecoration(
                color: getEmotionColor(
                  streak.currentStreakEmotion!,
                ).withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(8.r),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    AppAssets.getEmotionEmoji(streak.currentStreakEmotion!),
                    style: TextStyle(fontSize: 20.sp),
                  ),
                  SizedBox(width: 8.w),
                  CustomText(
                    'Current: ${AppAssets.getEmotionName(streak.currentStreakEmotion!)}',
                    fontSize: 12,
                    color: getEmotionColor(streak.currentStreakEmotion!),
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildStreakItem(
    BuildContext context,
    String label,
    String value,
    Color color,
  ) {
    return Column(
      children: [
        CustomText(
          value,
          fontSize: 28,
          fontWeight: FontWeight.bold,
          color: color,
        ),
        SizedBox(height: 4.h),
        CustomText(
          label,
          fontSize: 12,
          color:
              Theme.of(context).textTheme.bodySmall?.color ??
              AppColors.textSecondary,
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
