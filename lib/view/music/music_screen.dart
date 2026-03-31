import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import 'package:youtube_player_flutter/youtube_player_flutter.dart';
import '../../data/constants/colors.dart';
import '../../data/constants/video_urls.dart';
import '../../widgets/custom_text.dart';
import '../../widgets/custom_app_bar.dart';
import '../../widgets/emotion_video_player.dart';

/// Music for Stress Relief screen
/// Shows a curated list of music/meditation videos that can be played in-app.
class MusicScreen extends StatelessWidget {
  const MusicScreen({super.key});

  // Accent color for this screen — calm blue, distinct from other buttons
  static const Color _accent = AppColors.primaryBlue;

  @override
  Widget build(BuildContext context) {
    final videos = EmotionVideoUrls.calmingMusicVideos;

    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      appBar: const CustomAppBar(
        title: 'Music for Stress Relief',
        showBackButton: true,
      ),
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Subtitle banner
          Container(
            width: double.infinity,
            margin: EdgeInsets.fromLTRB(16.w, 16.h, 16.w, 0),
            padding: EdgeInsets.symmetric(horizontal: 16.w, vertical: 14.h),
            decoration: BoxDecoration(
              color: _accent.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(16.r),
              border: Border.all(color: _accent.withValues(alpha: 0.18)),
            ),
            child: Row(
              children: [
                Container(
                  padding: EdgeInsets.all(8.w),
                  decoration: BoxDecoration(
                    color: _accent.withValues(alpha: 0.15),
                    shape: BoxShape.circle,
                  ),
                  child: Icon(
                    Icons.music_note_rounded,
                    color: _accent,
                    size: 22.sp,
                  ),
                ),
                SizedBox(width: 12.w),
                Expanded(
                  child: CustomText(
                    'Curated music and meditation tracks to calm your mind and reduce stress.',
                    fontSize: 13,
                    color: AppColors.textSecondary,
                    height: 1.45,
                  ),
                ),
              ],
            ),
          ),

          SizedBox(height: 20.h),

          // Section header
          Padding(
            padding: EdgeInsets.symmetric(horizontal: 16.w),
            child: Row(
              children: [
                Icon(Icons.play_circle_fill, color: _accent, size: 22.sp),
                SizedBox(width: 8.w),
                CustomText(
                  'Suggested Videos',
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: Theme.of(context).textTheme.bodyLarge?.color,
                ),
              ],
            ),
          ),

          SizedBox(height: 12.h),

          // Video list or empty state
          Expanded(
            child: videos.isEmpty
                ? _buildEmptyState(context)
                : ListView.separated(
                    padding: EdgeInsets.fromLTRB(16.w, 0, 16.w, 32.h),
                    itemCount: videos.length,
                    separatorBuilder: (_, __) => SizedBox(height: 16.h),
                    itemBuilder: (context, index) =>
                        _buildVideoCard(context, videos[index]),
                  ),
          ),
        ],
      ),
    );
  }

  // ─── Video card ──────────────────────────────────────────────────────────────

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
          border: Border.all(color: _accent.withValues(alpha: 0.18)),
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
            // Thumbnail area — 16:9 aspect ratio
            AspectRatio(
              aspectRatio: 16 / 9,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  // Thumbnail image or fallback
                  if (hasThumbnail)
                    Image.network(
                      'https://img.youtube.com/vi/$videoId/hqdefault.jpg',
                      fit: BoxFit.cover,
                      loadingBuilder: (ctx, child, progress) {
                        if (progress == null) return child;
                        return Container(
                          color: _accent.withValues(alpha: 0.08),
                          child: Center(
                            child: SizedBox(
                              width: 28.w,
                              height: 28.w,
                              child: CircularProgressIndicator(
                                strokeWidth: 2.5,
                                color: _accent,
                              ),
                            ),
                          ),
                        );
                      },
                      errorBuilder: (ctx, err, _) => _buildThumbnailFallback(),
                    )
                  else
                    _buildThumbnailFallback(),

                  // Dark gradient overlay for readability
                  if (hasThumbnail)
                    Container(
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          begin: Alignment.topCenter,
                          end: Alignment.bottomCenter,
                          colors: [
                            Colors.transparent,
                            Colors.black.withValues(alpha: 0.25),
                          ],
                        ),
                      ),
                    ),

                  // Play button centered overlay
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

            // Title + tap hint
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
                    color: _accent,
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

  Widget _buildThumbnailFallback() {
    return Container(
      color: _accent.withValues(alpha: 0.08),
      child: Center(
        child: Icon(
          Icons.music_video_rounded,
          color: _accent.withValues(alpha: 0.5),
          size: 48.sp,
        ),
      ),
    );
  }

  // ─── Empty state ─────────────────────────────────────────────────────────────

  Widget _buildEmptyState(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            Icons.music_off_rounded,
            size: 64.sp,
            color: AppColors.textLight.withValues(alpha: 0.4),
          ),
          SizedBox(height: 16.h),
          CustomText(
            'No music videos yet',
            fontSize: 18,
            fontWeight: FontWeight.w600,
            color: AppColors.textSecondary,
          ),
          SizedBox(height: 8.h),
          CustomText(
            'Check back soon — new tracks are on the way!',
            fontSize: 14,
            color: AppColors.textLight,
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  // ─── Play dialog ─────────────────────────────────────────────────────────────

  void _playVideo(BuildContext context, String url, String title) {
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
              // Dialog header
              Padding(
                padding: EdgeInsets.fromLTRB(16.w, 16.h, 8.w, 8.h),
                child: Row(
                  children: [
                    Icon(Icons.music_note_rounded, color: _accent, size: 20.sp),
                    SizedBox(width: 8.w),
                    Expanded(
                      child: CustomText(
                        title,
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

  // ─── Helpers ─────────────────────────────────────────────────────────────────

  /// Extracts YouTube video ID from any standard YouTube URL.
  /// Returns empty string on failure — triggers thumbnail fallback.
  String _extractVideoId(String url) {
    if (url.isEmpty) return '';
    return YoutubePlayer.convertUrlToId(url) ?? '';
  }
}
