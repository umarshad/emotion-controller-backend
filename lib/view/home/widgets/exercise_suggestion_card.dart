import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import '../../../data/constants/colors.dart';
import '../../../data/controllers/emotion_controller.dart';
import '../../../widgets/custom_text.dart';
import '../../../widgets/custom_button.dart';

/// Exercise suggestion card based on current mood
class ExerciseSuggestionCard extends StatelessWidget {
  const ExerciseSuggestionCard({super.key});

  @override
  Widget build(BuildContext context) {
    final emotionController = Get.find<EmotionController>();

    return Obx(() {
      final recentEmotions = emotionController.getRecentEmotions(count: 1);
      final currentMood = recentEmotions.isNotEmpty ? recentEmotions.first : null;

      if (currentMood == null) {
        return const SizedBox.shrink();
      }

      final suggestions = _getExerciseSuggestions(currentMood.type);

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
                Icon(
                  Icons.fitness_center,
                  color: AppColors.primary,
                  size: 24.sp,
                ),
                SizedBox(width: 8.w),
                CustomText(
                  'Suggested Exercises',
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ],
            ),
            SizedBox(height: 16.h),
            ...suggestions.take(3).map((suggestion) => Padding(
                  padding: EdgeInsets.only(bottom: 12.h),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Container(
                        width: 6.w,
                        height: 6.w,
                        margin: EdgeInsets.only(top: 6.h, right: 12.w),
                        decoration: BoxDecoration(
                          color: AppColors.primary,
                          shape: BoxShape.circle,
                        ),
                      ),
                      Expanded(
                        child: CustomText(
                          suggestion,
                          fontSize: 14,
                          color: AppColors.textSecondary,
                        ),
                      ),
                    ],
                  ),
                )),
            SizedBox(height: 8.h),
            CustomButton(
              text: 'View All Suggestions',
              onPressed: () {
                // Navigate to detailed suggestions
              },
              isOutlined: true,
              backgroundColor: AppColors.primary,
              width: double.infinity,
            ),
          ],
        ),
      );
    });
  }

  List<String> _getExerciseSuggestions(String emotionType) {
    switch (emotionType.toLowerCase()) {
      case 'stress':
      case 'anxiety':
        return [
          'Practice deep breathing for 5 minutes',
          'Take a 10-minute walk outside',
          'Try progressive muscle relaxation',
          'Listen to calming music',
        ];
      case 'sadness':
      case 'loneliness':
        return [
          'Connect with a friend or loved one',
          'Engage in a creative activity',
          'Practice gratitude journaling',
          'Do something kind for yourself',
        ];
      case 'anger':
        return [
          'Go for a brisk walk or run',
          'Practice mindfulness meditation',
          'Write in a journal',
          'Try physical exercise',
        ];
      case 'tired':
        return [
          'Take a short power nap (20 minutes)',
          'Do gentle stretching',
          'Get some fresh air',
          'Stay hydrated',
        ];
      case 'happiness':
      case 'calm':
      case 'relaxed':
        return [
          'Maintain this positive state',
          'Share your joy with others',
          'Continue mindfulness practices',
          'Engage in activities you enjoy',
        ];
      default:
        return [
          'Take a moment to breathe',
          'Practice self-compassion',
          'Engage in a favorite activity',
          'Reach out for support if needed',
        ];
    }
  }
}
