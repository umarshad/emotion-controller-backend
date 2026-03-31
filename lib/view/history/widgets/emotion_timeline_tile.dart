import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import '../../../data/constants/colors.dart';
import '../../../data/constants/assets.dart';
import '../../../data/models/emotion_model.dart';
import '../../../data/utils/helpers.dart';
import '../../../widgets/custom_text.dart';

/// Individual emotion entry in timeline
class EmotionTimelineTile extends StatelessWidget {
  final EmotionModel emotion;
  final bool isFirst;
  final bool isLast;

  const EmotionTimelineTile({
    super.key,
    required this.emotion,
    this.isFirst = false,
    this.isLast = false,
  });

  @override
  Widget build(BuildContext context) {
    final emotionColor = getEmotionColor(emotion.type);
    final emoji = AppAssets.getEmotionEmoji(emotion.type);
    final emotionName = AppAssets.getEmotionName(emotion.type);

    return IntrinsicHeight(
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Timeline line
          Column(
            children: [
              if (!isFirst)
                Container(
                  width: 2.w,
                  height: 20.h,
                  color: Theme.of(context).dividerColor,
                ),
              Container(
                width: 16.w,
                height: 16.h,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: emotionColor,
                  border: Border.all(
                    color: Theme.of(context).scaffoldBackgroundColor,
                    width: 3.w,
                  ),
                ),
              ),
              if (!isLast)
                Expanded(
                  child: Container(
                    width: 2.w,
                    color: Theme.of(context).dividerColor,
                  ),
                ),
            ],
          ),
          SizedBox(width: 16.w),

          // Emotion card
          Expanded(
            child: Container(
              margin: EdgeInsets.only(bottom: 16.h),
              padding: EdgeInsets.all(16.w),
              decoration: BoxDecoration(
                color: Theme.of(context).cardColor,
                borderRadius: BorderRadius.circular(12.r),
                border: Border.all(
                  color: emotionColor.withValues(alpha: 0.3),
                  width: 1.5.w,
                ),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Emoji
                      Container(
                        width: 40.w,
                        height: 40.h,
                        decoration: BoxDecoration(
                          color: emotionColor.withValues(alpha: 0.1),
                          borderRadius: BorderRadius.circular(10.r),
                        ),
                        child: Center(
                          child: Text(emoji, style: TextStyle(fontSize: 24.sp)),
                        ),
                      ),
                      SizedBox(width: 12.w),
                      // Emotion info
                      Expanded(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            CustomText(
                              emotionName,
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                              color: emotionColor,
                            ),
                            SizedBox(height: 4.h),
                            CustomText(
                              '${emotion.intensity}% • ${Helpers.getIntensityLevel(emotion.intensity)}',
                              fontSize: 12,
                              color:
                                  Theme.of(
                                    context,
                                  ).textTheme.bodySmall?.color ??
                                  AppColors.textSecondary,
                            ),
                          ],
                        ),
                      ),
                      // Method indicator
                      Container(
                        padding: EdgeInsets.symmetric(
                          horizontal: 8.w,
                          vertical: 4.h,
                        ),
                        decoration: BoxDecoration(
                          color: emotion.detectionMethod == 'chat'
                              ? AppColors.primary.withValues(alpha: 0.1)
                              : AppColors.secondary.withValues(alpha: 0.1),
                          borderRadius: BorderRadius.circular(6.r),
                        ),
                        child: CustomText(
                          emotion.detectionMethod == 'chat' ? '💬' : '📷',
                          fontSize: 14,
                        ),
                      ),
                    ],
                  ),
                  SizedBox(height: 12.h),
                  // Intensity bar
                  ClipRRect(
                    borderRadius: BorderRadius.circular(4.r),
                    child: LinearProgressIndicator(
                      value: emotion.intensity / 100,
                      backgroundColor: emotionColor.withValues(alpha: 0.1),
                      valueColor: AlwaysStoppedAnimation<Color>(emotionColor),
                      minHeight: 6.h,
                    ),
                  ),
                  SizedBox(height: 8.h),
                  // Timestamp
                  CustomText(
                    Helpers.formatDateTime(emotion.timestamp),
                    fontSize: 11,
                    color:
                        Theme.of(context).textTheme.bodySmall?.color ??
                        AppColors.textLight,
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
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
