import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import '../data/constants/colors.dart' as colors;
import '../data/constants/assets.dart';
import '../data/models/emotion_model.dart';
import '../data/utils/helpers.dart';
import 'custom_text.dart';
import 'custom_button.dart';
import 'animated_emotion_widget.dart';

/// Alert dialogs and bottom sheets for emotion results
class ReusableDialogs {
  /// Show emotion detection result dialog
  static void showEmotionResultDialog(EmotionModel emotion) {
    Get.dialog(
      Dialog(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20.r),
        ),
        child: Container(
          padding: EdgeInsets.all(24.w),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Emotion emoji and name
              AnimatedEmotionWidget(emotionType: emotion.type, size: 80),
              SizedBox(height: 16.h),
              CustomText(
                AppAssets.getEmotionName(emotion.type),
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: colors.getEmotionColor(emotion.type),
              ),
              SizedBox(height: 8.h),
              CustomText(
                '${emotion.intensity}% ${Helpers.getIntensityLevel(emotion.intensity)}',
                fontSize: 16,
                color: colors.AppColors.textSecondary,
              ),
              SizedBox(height: 24.h),
              // Intensity bar
              ClipRRect(
                borderRadius: BorderRadius.circular(8.r),
                child: LinearProgressIndicator(
                  value: emotion.intensity / 100,
                  backgroundColor: colors
                      .getEmotionColor(emotion.type)
                      .withValues(alpha: 0.1),
                  valueColor: AlwaysStoppedAnimation<Color>(
                    colors.getEmotionColor(emotion.type),
                  ),
                  minHeight: 8.h,
                ),
              ),
              SizedBox(height: 24.h),
              // Suggestions
              Align(
                alignment: Alignment.centerLeft,
                child: CustomText(
                  'Suggestions:',
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
              SizedBox(height: 12.h),
              ...emotion.suggestions
                  .take(3)
                  .map(
                    (suggestion) => Padding(
                      padding: EdgeInsets.only(bottom: 8.h),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Icon(
                            Icons.check_circle_outline,
                            size: 20.sp,
                            color: colors.getEmotionColor(emotion.type),
                          ),
                          SizedBox(width: 8.w),
                          Expanded(
                            child: CustomText(
                              suggestion,
                              fontSize: 14,
                              color: colors.AppColors.textSecondary,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
              SizedBox(height: 24.h),
              CustomButton(
                text: 'Got it',
                onPressed: () => Get.back(),
                width: double.infinity,
              ),
            ],
          ),
        ),
      ),
      barrierDismissible: true,
    );
  }

  /// Show bottom sheet with emotion details
  static void showEmotionBottomSheet(EmotionModel emotion) {
    Get.bottomSheet(
      Container(
        padding: EdgeInsets.all(24.w),
        decoration: BoxDecoration(
          color: colors.AppColors.cardBackground,
          borderRadius: BorderRadius.only(
            topLeft: Radius.circular(24.r),
            topRight: Radius.circular(24.r),
          ),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Handle
            Container(
              width: 40.w,
              height: 4.h,
              margin: EdgeInsets.only(bottom: 24.h),
              decoration: BoxDecoration(
                color: colors.AppColors.divider,
                borderRadius: BorderRadius.circular(2.r),
              ),
            ),
            // Emotion display
            AnimatedEmotionWidget(emotionType: emotion.type, size: 100),
            SizedBox(height: 16.h),
            CustomText(
              AppAssets.getEmotionName(emotion.type),
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: colors.getEmotionColor(emotion.type),
            ),
            SizedBox(height: 8.h),
            CustomText(
              Helpers.formatDateTime(emotion.timestamp),
              fontSize: 14,
              color: colors.AppColors.textSecondary,
            ),
            SizedBox(height: 24.h),
            // All suggestions
            Align(
              alignment: Alignment.centerLeft,
              child: CustomText(
                'Personalized Suggestions:',
                fontSize: 18,
                fontWeight: FontWeight.w600,
              ),
            ),
            SizedBox(height: 16.h),
            ...emotion.suggestions.map(
              (suggestion) => Padding(
                padding: EdgeInsets.only(bottom: 12.h),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(
                      Icons.lightbulb_outline,
                      size: 20.sp,
                      color: colors.getEmotionColor(emotion.type),
                    ),
                    SizedBox(width: 12.w),
                    Expanded(
                      child: CustomText(
                        suggestion,
                        fontSize: 14,
                        color: colors.AppColors.textSecondary,
                      ),
                    ),
                  ],
                ),
              ),
            ),
            SizedBox(height: 24.h),
          ],
        ),
      ),
      isScrollControlled: true,
      enableDrag: true,
    );
  }
}
