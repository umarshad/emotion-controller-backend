import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import '../../../data/constants/colors.dart';
import '../../../data/controllers/emotion_controller.dart';
import '../../../data/controllers/navigation_controller.dart';
import '../../../widgets/custom_text.dart';
import '../../history/widgets/detailed_emotion_view.dart';
import 'emotion_chip.dart';

/// Recent emotions overview card (horizontal scroll)
class EmotionOverviewCard extends StatelessWidget {
  const EmotionOverviewCard({super.key});

  @override
  Widget build(BuildContext context) {
    final emotionController = Get.find<EmotionController>();

    return Obx(() {
      // Fetch 4 to know if we have more than 3 (for View All button)
      final allRecent = emotionController.getRecentEmotions(count: 4);
      final recentEmotions = allRecent.take(3).toList();
      final hasMore = allRecent.length > 3;

      if (recentEmotions.isEmpty) {
        // Empty state handled cleanly in parent or just hidden/minimal
        return const SizedBox.shrink();
      }

      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              CustomText(
                'Recent', // Shortened title
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: AppColors.textPrimary,
                letterSpacing: -0.5,
              ),
              if (hasMore)
                TextButton(
                  onPressed: () {
                    Get.find<NavigationController>().changeIndex(3);
                  },
                  child: CustomText(
                    'See All',
                    fontSize: 14,
                    color: AppColors.primary,
                    fontWeight: FontWeight.w600,
                  ),
                ),
            ],
          ),
          SizedBox(height: 12.h),
          SizedBox(
            height: 140.h, // Increased height to prevent overflow (was 100.h)
            child: ListView.separated(
              scrollDirection: Axis.horizontal,
              itemCount: recentEmotions.length,
              separatorBuilder: (context, index) => SizedBox(width: 12.w),
              itemBuilder: (context, index) {
                final emotion = recentEmotions[index];
                // Use a modified EmotionChip or wrap it for cleaner look
                return EmotionChip(
                  emotion: emotion,
                  onTap: () {
                    Get.to(() => DetailedEmotionView(emotion: emotion));
                  },
                );
              },
            ),
          ),
        ],
      );
    });
  }
}
