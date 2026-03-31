import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import '../../../data/constants/colors.dart';
import '../../../data/constants/assets.dart';
import '../../../data/controllers/emotion_controller.dart';
import '../../../widgets/custom_text.dart';
import '../../../widgets/animated_emotion_widget.dart';

/// Current mood card with large emotion display (minimal design)
class CurrentMoodCard extends StatelessWidget {
  final VoidCallback? onCheckIn;

  const CurrentMoodCard({super.key, this.onCheckIn});

  @override
  Widget build(BuildContext context) {
    final emotionController = Get.find<EmotionController>();

    return Obx(() {
      final recentEmotions = emotionController.getRecentEmotions(count: 1);
      final currentMood = recentEmotions.isNotEmpty
          ? recentEmotions.first
          : null;

      return Container(
        margin: EdgeInsets.symmetric(horizontal: 16.w),
        padding: EdgeInsets.symmetric(vertical: 24.h, horizontal: 20.w),
        decoration: BoxDecoration(
          // Ultra-soft colorful background based on mood
          color: currentMood != null
              ? getEmotionColor(currentMood.type).withValues(alpha: 0.08)
              : AppColors.cardBackground,
          borderRadius: BorderRadius.circular(32.r), // High rounding
          // No shadows for flat minimal look
        ),
        child: Column(
          children: [
            if (currentMood != null) ...[
              // Minimalistic Top Row
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  AnimatedEmotionWidget(
                    emotionType: currentMood.type,
                    size: 80,
                  ),
                  SizedBox(width: 24.w),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      CustomText(
                        'Current Mood',
                        fontSize: 14,
                        color: AppColors.textSecondary,
                        fontWeight: FontWeight.w500,
                        letterSpacing: 0.5,
                      ),
                      SizedBox(height: 4.h),
                      CustomText(
                        AppAssets.getEmotionName(currentMood.type),
                        fontSize: 32,
                        fontWeight:
                            FontWeight.w300, // Thinner font for elegance
                        color: getEmotionColor(currentMood.type),
                      ),
                      SizedBox(height: 4.h),
                      Container(
                        padding: EdgeInsets.symmetric(
                          horizontal: 8.w,
                          vertical: 4.h,
                        ),
                        decoration: BoxDecoration(
                          color: getEmotionColor(
                            currentMood.type,
                          ).withValues(alpha: 0.2),
                          borderRadius: BorderRadius.circular(12.r),
                        ),
                        child: CustomText(
                          '${currentMood.intensity}% Intense',
                          fontSize: 12,
                          color: getEmotionColor(
                            currentMood.type,
                          ).withValues(alpha: 0.9),
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ] else ...[
              // Empty State - Clean and simple
              Column(
                children: [
                  Icon(
                    Icons.sentiment_neutral,
                    size: 60.sp,
                    color: AppColors.textLight.withValues(alpha: 0.5),
                  ),
                  SizedBox(height: 16.h),
                  CustomText(
                    'No mood detected yet',
                    fontSize: 18,
                    fontWeight: FontWeight.w500,
                    color: AppColors.textSecondary,
                  ),
                  SizedBox(height: 8.h),
                  CustomText(
                    'Tap check-in to start tracking',
                    fontSize: 14,
                    color: AppColors.textLight,
                  ),
                ],
              ),
            ],
            SizedBox(height: 24.h),
            // Minimalistic Button
            SizedBox(
              width: double.infinity,
              height: 50.h,
              child: ElevatedButton(
                onPressed: onCheckIn,
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.primary,
                  elevation: 0, // Flat button
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16.r),
                  ),
                ),
                child: CustomText(
                  currentMood != null ? 'Check In Again' : 'Check In Mood',
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
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
