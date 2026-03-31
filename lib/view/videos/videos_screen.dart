import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import '../../data/constants/colors.dart';
import '../../data/constants/video_urls.dart';
import '../../widgets/custom_text.dart';
import '../../widgets/custom_app_bar.dart';
import 'video_category_screen.dart';

/// Videos hub — 5 category tiles in a 2-column grid.
/// Each tile navigates to VideoCategoryScreen with the relevant video list.
class VideosScreen extends StatelessWidget {
  const VideosScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final categories = _buildCategories();

    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      appBar: const CustomAppBar(title: 'Videos', showBackButton: true),
      body: SingleChildScrollView(
        padding: EdgeInsets.fromLTRB(16.w, 20.h, 16.w, 32.h),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header description
            Container(
              width: double.infinity,
              padding: EdgeInsets.symmetric(horizontal: 16.w, vertical: 14.h),
              decoration: BoxDecoration(
                color: AppColors.primary.withValues(alpha: 0.07),
                borderRadius: BorderRadius.circular(16.r),
                border: Border.all(
                  color: AppColors.primary.withValues(alpha: 0.15),
                ),
              ),
              child: Row(
                children: [
                  Container(
                    padding: EdgeInsets.all(8.w),
                    decoration: BoxDecoration(
                      color: AppColors.primary.withValues(alpha: 0.15),
                      shape: BoxShape.circle,
                    ),
                    child: Icon(
                      Icons.play_circle_filled,
                      color: AppColors.primary,
                      size: 22.sp,
                    ),
                  ),
                  SizedBox(width: 12.w),
                  Expanded(
                    child: CustomText(
                      'Curated videos to support your emotional wellbeing — music, Quran, laughter, and more.',
                      fontSize: 13,
                      color: AppColors.textSecondary,
                      height: 1.45,
                    ),
                  ),
                ],
              ),
            ),

            SizedBox(height: 24.h),

            // Category grid
            GridView.builder(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2,
                crossAxisSpacing: 14.w,
                mainAxisSpacing: 14.h,
                childAspectRatio: 1.05,
              ),
              itemCount: categories.length,
              itemBuilder: (context, index) =>
                  _buildCategoryCard(context, categories[index]),
            ),
          ],
        ),
      ),
    );
  }

  // ─── Category definitions ─────────────────────────────────────────────────

  List<_VideoCategory> _buildCategories() => [
    _VideoCategory(
      title: 'Calming Music',
      subtitle: '${EmotionVideoUrls.calmingMusicVideos.length} videos',
      icon: Icons.music_note_rounded,
      color: AppColors.primaryBlue,
      videos: EmotionVideoUrls.calmingMusicVideos,
    ),
    _VideoCategory(
      title: 'Quran',
      subtitle: '${EmotionVideoUrls.quranVideos.length} videos',
      icon: Icons.menu_book_rounded,
      color: AppColors.primary, // teal
      videos: EmotionVideoUrls.quranVideos,
    ),
    _VideoCategory(
      title: 'Funny Videos',
      subtitle: '${EmotionVideoUrls.funnyVideos.length} videos',
      icon: Icons.sentiment_very_satisfied_rounded,
      color: AppColors.emotionHappy,
      videos: EmotionVideoUrls.funnyVideos,
    ),
    _VideoCategory(
      title: 'Calming',
      subtitle: '${EmotionVideoUrls.calmingVideos.length} video',
      icon: Icons.self_improvement_rounded,
      color: AppColors.secondary, // lavender
      videos: EmotionVideoUrls.calmingVideos,
    ),
    _VideoCategory(
      title: 'Fitness',
      subtitle: '${EmotionVideoUrls.fitnessVideos.length} videos',
      icon: Icons.fitness_center_rounded,
      color: AppColors.accentCoral,
      videos: EmotionVideoUrls.fitnessVideos,
    ),
  ];

  // ─── Category card ────────────────────────────────────────────────────────

  Widget _buildCategoryCard(BuildContext context, _VideoCategory cat) {
    return GestureDetector(
      onTap: () => Get.to(
        () => VideoCategoryScreen(
          title: cat.title,
          accentColor: cat.color,
          icon: cat.icon,
          videos: cat.videos,
        ),
      ),
      child: Container(
        decoration: BoxDecoration(
          color: Theme.of(context).cardColor,
          borderRadius: BorderRadius.circular(20.r),
          border: Border.all(color: cat.color.withValues(alpha: 0.2)),
          boxShadow: [
            BoxShadow(
              color: cat.color.withValues(alpha: 0.06),
              blurRadius: 12,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Icon badge
            Container(
              padding: EdgeInsets.all(16.w),
              decoration: BoxDecoration(
                color: cat.color.withValues(alpha: 0.12),
                shape: BoxShape.circle,
              ),
              child: Icon(cat.icon, color: cat.color, size: 30.sp),
            ),
            SizedBox(height: 12.h),

            // Title
            CustomText(
              cat.title,
              fontSize: 15,
              fontWeight: FontWeight.bold,
              color:
                  Theme.of(context).textTheme.bodyLarge?.color ??
                  AppColors.textPrimary,
              textAlign: TextAlign.center,
            ),
            SizedBox(height: 4.h),

            // Subtitle — video count
            CustomText(
              cat.subtitle,
              fontSize: 12,
              color: cat.color,
              fontWeight: FontWeight.w500,
            ),
          ],
        ),
      ),
    );
  }
}

// ─── Internal category model ───────────────────────────────────────────────

class _VideoCategory {
  final String title;
  final String subtitle;
  final IconData icon;
  final Color color;
  final List<Map<String, String>> videos;

  const _VideoCategory({
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.color,
    required this.videos,
  });
}
