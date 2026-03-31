import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import '../data/constants/colors.dart';
import '../data/constants/assets.dart';
import '../data/models/emotion_model.dart';
import '../data/utils/helpers.dart';
import 'custom_text.dart';

/// Card displaying emotion with icon, name, intensity
class EmotionCard extends StatelessWidget {
  final EmotionModel emotion;
  final VoidCallback? onTap;
  final bool showDetails;

  const EmotionCard({
    super.key,
    required this.emotion,
    this.onTap,
    this.showDetails = false,
  });

  @override
  Widget build(BuildContext context) {
    final emotionColor = getEmotionColor(emotion.type);
    final emoji = AppAssets.getEmotionEmoji(emotion.type);
    final emotionName = AppAssets.getEmotionName(emotion.type);

    return GestureDetector(
      onTap: onTap,
      child: Container(
        margin: EdgeInsets.only(bottom: 12.h),
        padding: EdgeInsets.all(16.w),
        decoration: BoxDecoration(
          color: Theme.of(context).cardColor,
          borderRadius: BorderRadius.circular(16.r),
          border: Border.all(
            color: emotionColor.withValues(alpha: 0.3),
            width: 1.5.w,
          ),
          boxShadow: [
            BoxShadow(
              color: emotionColor.withValues(alpha: 0.1),
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
                // Emoji
                Container(
                  width: 50.w,
                  height: 50.h,
                  decoration: BoxDecoration(
                    color: emotionColor.withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(12.r),
                  ),
                  child: Center(
                    child: Text(emoji, style: TextStyle(fontSize: 28.sp)),
                  ),
                ),
                SizedBox(width: 12.w),
                // Emotion info
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      CustomText(
                        emotionName,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: emotionColor,
                      ),
                      SizedBox(height: 4.h),
                      Row(
                        children: [
                          CustomText(
                            '${emotion.intensity}%',
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                            color:
                                Theme.of(context).textTheme.bodySmall?.color ??
                                AppColors.textSecondary,
                          ),
                          SizedBox(width: 8.w),
                          Container(
                            padding: EdgeInsets.symmetric(
                              horizontal: 8.w,
                              vertical: 2.h,
                            ),
                            decoration: BoxDecoration(
                              color: emotionColor.withValues(alpha: 0.1),
                              borderRadius: BorderRadius.circular(8.r),
                            ),
                            child: CustomText(
                              Helpers.getIntensityLevel(emotion.intensity),
                              fontSize: 11,
                              color: emotionColor,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
                // Method indicator
                Container(
                  padding: EdgeInsets.symmetric(
                    horizontal: 10.w,
                    vertical: 6.h,
                  ),
                  decoration: BoxDecoration(
                    color: emotion.detectionMethod == 'chat'
                        ? AppColors.primary.withValues(alpha: 0.1)
                        : AppColors.secondary.withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(8.r),
                  ),
                  child: CustomText(
                    emotion.detectionMethod == 'chat' ? '💬' : '📷',
                    fontSize: 16,
                  ),
                ),
              ],
            ),
            if (showDetails) ...[
              SizedBox(height: 12.h),
              Divider(color: Theme.of(context).dividerColor),
              SizedBox(height: 12.h),
              CustomText(
                Helpers.formatDateTime(emotion.timestamp),
                fontSize: 12,
                color:
                    Theme.of(context).textTheme.bodySmall?.color ??
                    AppColors.textSecondary,
              ),
              SizedBox(height: 8.h),
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
            ],
          ],
        ),
      ),
    );
  }
}
