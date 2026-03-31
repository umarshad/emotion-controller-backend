import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import '../../../data/constants/colors.dart';
import '../../../data/constants/assets.dart';
import '../../../data/models/emotion_model.dart';
import '../../../data/utils/helpers.dart';
import '../../../widgets/custom_text.dart';
import '../../../widgets/animated_emotion_widget.dart';

import 'package:get/get.dart';
import '../../history/widgets/detailed_emotion_view.dart' hide getEmotionColor;

/// Widget displaying detected emotion, intensity, and suggestions
class EmotionResponseWidget extends StatelessWidget {
  final EmotionModel emotion;

  const EmotionResponseWidget({super.key, required this.emotion});

  @override
  Widget build(BuildContext context) {
    final emotionColor = getEmotionColor(emotion.type);
    final emotionName = AppAssets.getEmotionName(emotion.type);

    return Container(
      margin: EdgeInsets.symmetric(horizontal: 16.w, vertical: 8.h),
      padding: EdgeInsets.all(16.w),
      decoration: BoxDecoration(
        color: emotionColor.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(16.r),
        border: Border.all(color: emotionColor.withValues(alpha: 0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              AnimatedEmotionWidget(emotionType: emotion.type, size: 50),
              SizedBox(width: 12.w),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    CustomText(
                      'Emotion Detected: $emotionName',
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: emotionColor,
                    ),
                    SizedBox(height: 4.h),
                    CustomText(
                      'Intensity: ${emotion.intensity}% (${Helpers.getIntensityLevel(emotion.intensity)})',
                      fontSize: 12,
                      color:
                          Theme.of(context).textTheme.bodySmall?.color ??
                          AppColors.textSecondary,
                    ),
                  ],
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
              backgroundColor: emotionColor.withValues(alpha: 0.2),
              valueColor: AlwaysStoppedAnimation<Color>(emotionColor),
              minHeight: 6.h,
            ),
          ),
          SizedBox(height: 12.h),
          // Suggestions preview
          CustomText('Suggestions:', fontSize: 14, fontWeight: FontWeight.w600),
          SizedBox(height: 8.h),
          ...emotion.suggestions
              .take(2)
              .map(
                (suggestion) => Padding(
                  padding: EdgeInsets.only(bottom: 4.h),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Icon(
                        Icons.check_circle_outline,
                        size: 16.sp,
                        color: emotionColor,
                      ),
                      SizedBox(width: 8.w),
                      Expanded(
                        child: CustomText(
                          suggestion,
                          fontSize: 12,
                          color:
                              Theme.of(context).textTheme.bodySmall?.color ??
                              AppColors.textSecondary,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
          SizedBox(height: 8.h),
          GestureDetector(
            onTap: () {
              Get.to(() => DetailedEmotionView(emotion: emotion));
            },
            child: Container(
              padding: EdgeInsets.symmetric(vertical: 8.h),
              child: CustomText(
                'View all suggestions →',
                fontSize: 12,
                color: emotionColor,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
