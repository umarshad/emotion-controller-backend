import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import '../../../data/constants/colors.dart';
import '../../../data/constants/assets.dart';
import '../../../widgets/custom_text.dart';

/// Emotion filter chip widget
class EmotionFilterChip extends StatelessWidget {
  final String emotionType;
  final bool isSelected;
  final VoidCallback onTap;

  const EmotionFilterChip({
    super.key,
    required this.emotionType,
    required this.isSelected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    if (emotionType == 'all') {
      return GestureDetector(
        onTap: onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          padding: EdgeInsets.symmetric(horizontal: 12.w, vertical: 8.h),
          decoration: BoxDecoration(
            color: isSelected ? AppColors.primary : Theme.of(context).cardColor,
            borderRadius: BorderRadius.circular(20.r),
            border: Border.all(
              color: isSelected
                  ? AppColors.primary
                  : Theme.of(context).dividerColor,
              width: isSelected ? 2.w : 1.w,
            ),
          ),
          child: CustomText(
            'All Emotions',
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: isSelected
                ? Colors.white
                : (Theme.of(context).textTheme.bodyLarge?.color ??
                      AppColors.textPrimary),
          ),
        ),
      );
    }

    final emotionColor = getEmotionColor(emotionType);
    final emoji = AppAssets.getEmotionEmoji(emotionType);
    final emotionName = AppAssets.getEmotionName(emotionType);

    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        constraints: BoxConstraints(minWidth: 90.w),
        padding: EdgeInsets.symmetric(horizontal: 12.w, vertical: 10.h),
        decoration: BoxDecoration(
          color: isSelected
              ? emotionColor
              : emotionColor.withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(20.r),
          border: Border.all(
            color: isSelected
                ? emotionColor
                : emotionColor.withValues(alpha: 0.3),
            width: isSelected ? 2.w : 1.w,
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(emoji, style: TextStyle(fontSize: 16.sp)),
            SizedBox(width: 6.w),
            CustomText(
              emotionName,
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: isSelected ? Colors.white : emotionColor,
            ),
          ],
        ),
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
