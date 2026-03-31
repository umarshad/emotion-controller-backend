import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import 'package:youtube_player_flutter/youtube_player_flutter.dart';
import '../../data/constants/colors.dart';
import '../../widgets/custom_text.dart';
import '../../widgets/custom_app_bar.dart';
import '../../widgets/emotion_video_player.dart';

/// Reusable video list screen for any category.
/// Pass [title], [accentColor], [icon], and [videos] from the parent.
class VideoCategoryScreen extends StatelessWidget {
  final String title;
  final Color accentColor;
  final IconData icon;
  final List<Map<String, String>> videos;

  const VideoCategoryScreen({
    super.key,
    required this.title,
    required this.accentColor,
    required this.icon,
    required this.videos,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      appBar: CustomAppBar(title: title, showBackButton: true),
      body: videos.isEmpty
          ? _buildEmptyState(context)
          : ListView.separated(
              padding: EdgeInsets.fromLTRB(16.w, 20.h, 16.w, 32.h),
              itemCount: videos.length,
              separatorBuilder: (_, __) => SizedBox(height: 16.h),
              itemBuilder: (context, index) =>
                  _buildVideoCard(context, videos[index]),
            ),
    );
  }

  // ─── Video card ─────────────────────────────────────────────────────────

  Widget _buildVideoCard(BuildContext context, Map<String, String> video) {
    final url = video['url'] ?? '';
    final title = video['title'] ?? '';
    final videoId = _extractVideoId(url);
    final hasThumbnail = videoId.isNotEmpty;

    return GestureDetector(
      onTap: () => _playVideo(context, url, title),
      child: Container(
        decoration: BoxDecoration(
          color: Theme.of(context).cardColor,
          borderRadius: BorderRadius.circular(16.r),
          border: Border.all(color: accentColor.withValues(alpha: 0.2)),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.04),
              blurRadius: 8,
              offset: const Offset(0, 3),
            ),
          ],
        ),
        clipBehavior: Clip.antiAlias,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // 16:9 thumbnail area
            AspectRatio(
              aspectRatio: 16 / 9,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  // Thumbnail or fallback
                  if (hasThumbnail)
                    Image.network(
                      'https://img.youtube.com/vi/$videoId/hqdefault.jpg',
                      fit: BoxFit.cover,
                      loadingBuilder: (ctx, child, progress) {
                        if (progress == null) return child;
                        return Container(
                          color: accentColor.withValues(alpha: 0.08),
                          child: Center(
                            child: SizedBox(
                              width: 28.w,
                              height: 28.w,
                              child: CircularProgressIndicator(
                                strokeWidth: 2.5,
                                color: accentColor,
                              ),
                            ),
                          ),
                        );
                      },
                      errorBuilder: (ctx, err, _) =>
                          _buildThumbnailFallback(context),
                    )
                  else
                    _buildThumbnailFallback(context),

                  // Gradient overlay
                  if (hasThumbnail)
                    Container(
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          begin: Alignment.topCenter,
                          end: Alignment.bottomCenter,
                          colors: [
                            Colors.transparent,
                            Colors.black.withValues(alpha: 0.28),
                          ],
                        ),
                      ),
                    ),

                  // Play button overlay
                  Center(
                    child: Container(
                      padding: EdgeInsets.all(10.w),
                      decoration: BoxDecoration(
                        color: Colors.black.withValues(alpha: 0.45),
                        shape: BoxShape.circle,
                      ),
                      child: Icon(
                        Icons.play_arrow_rounded,
                        color: Colors.white,
                        size: 32.sp,
                      ),
                    ),
                  ),
                ],
              ),
            ),

            // Title row
            Padding(
              padding: EdgeInsets.symmetric(horizontal: 14.w, vertical: 12.h),
              child: Row(
                children: [
                  Expanded(
                    child: CustomText(
                      title,
                      fontSize: 15,
                      fontWeight: FontWeight.w600,
                      color:
                          Theme.of(context).textTheme.bodyLarge?.color ??
                          AppColors.textPrimary,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      height: 1.35,
                    ),
                  ),
                  SizedBox(width: 8.w),
                  Icon(
                    Icons.play_circle_outline_rounded,
                    color: accentColor,
                    size: 22.sp,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildThumbnailFallback(BuildContext context) {
    return Container(
      color: accentColor.withValues(alpha: 0.08),
      child: Center(
        child: Icon(
          icon,
          color: accentColor.withValues(alpha: 0.45),
          size: 48.sp,
        ),
      ),
    );
  }

  // ─── Empty state ────────────────────────────────────────────────────────

  Widget _buildEmptyState(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            icon,
            size: 64.sp,
            color: AppColors.textLight.withValues(alpha: 0.35),
          ),
          SizedBox(height: 16.h),
          CustomText(
            'No videos yet',
            fontSize: 18,
            fontWeight: FontWeight.w600,
            color: AppColors.textSecondary,
          ),
          SizedBox(height: 8.h),
          CustomText(
            'Check back soon!',
            fontSize: 14,
            color: AppColors.textLight,
          ),
        ],
      ),
    );
  }

  // ─── Play dialog ────────────────────────────────────────────────────────

  void _playVideo(BuildContext context, String url, String videoTitle) {
    if (url.isEmpty) return;

    Get.dialog(
      Dialog(
        backgroundColor: Colors.transparent,
        insetPadding: EdgeInsets.all(16.w),
        child: Container(
          decoration: BoxDecoration(
            color: Theme.of(context).scaffoldBackgroundColor,
            borderRadius: BorderRadius.circular(20.r),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Header
              Padding(
                padding: EdgeInsets.fromLTRB(16.w, 14.h, 8.w, 8.h),
                child: Row(
                  children: [
                    Icon(icon, color: accentColor, size: 20.sp),
                    SizedBox(width: 8.w),
                    Expanded(
                      child: CustomText(
                        videoTitle,
                        fontSize: 15,
                        fontWeight: FontWeight.bold,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                    IconButton(
                      icon: const Icon(Icons.close),
                      visualDensity: VisualDensity.compact,
                      onPressed: () => Get.back(),
                    ),
                  ],
                ),
              ),
              // Embedded player
              ClipRRect(
                borderRadius: BorderRadius.only(
                  bottomLeft: Radius.circular(20.r),
                  bottomRight: Radius.circular(20.r),
                ),
                child: EmotionVideoPlayer(emotionType: '', videoUrl: url),
              ),
            ],
          ),
        ),
      ),
      barrierDismissible: true,
    );
  }

  // ─── Helpers ────────────────────────────────────────────────────────────

  String _extractVideoId(String url) {
    if (url.isEmpty) return '';
    return YoutubePlayer.convertUrlToId(url) ?? '';
  }
}
