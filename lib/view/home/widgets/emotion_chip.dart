import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import '../../../data/constants/colors.dart';
import '../../../data/constants/assets.dart';
import '../../../data/models/emotion_model.dart';
import '../../../widgets/custom_text.dart';

/// Minimal emotion chip widget for horizontal scrolling
class EmotionChip extends StatelessWidget {
  final EmotionModel emotion;
  final VoidCallback? onTap;

  const EmotionChip({super.key, required this.emotion, this.onTap});

  @override
  Widget build(BuildContext context) {
    final emotionColor = _getEmotionColor(emotion.type);
    final emoji = AppAssets.getEmotionEmoji(emotion.type);
    final emotionName = AppAssets.getEmotionName(emotion.type);

    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 100.w, // Reduced width as requested
        padding: EdgeInsets.symmetric(
          horizontal: 10.w,
          vertical: 12.h, // Optimized padding
        ),
        decoration: BoxDecoration(
          color: AppColors.cardBackground,
          borderRadius: BorderRadius.circular(20.r), // Slightly rounder
          border: Border.all(
            color: emotionColor.withValues(alpha: 0.3),
            width: 1.5.w,
          ),
          boxShadow: [
            BoxShadow(
              color: emotionColor.withValues(alpha: 0.08), // More subtle shadow
              blurRadius: 10,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(emoji, style: TextStyle(fontSize: 28.sp)), // Reduced from 32
            SizedBox(height: 5.h), // Reduced from 6.h (-1px)
            CustomText(
              emotionName,
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: emotionColor,
              textAlign: TextAlign.center,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              height: 1.2, // Consistent line height
            ),
            SizedBox(height: 1.h), // Reduced from 2.h (-1px)
            CustomText(
              '${emotion.intensity}%',
              fontSize: 10,
              color: AppColors.textSecondary,
            ),
          ],
        ),
      ),
    );
  }

  Color _getEmotionColor(String emotionType) {
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
}
