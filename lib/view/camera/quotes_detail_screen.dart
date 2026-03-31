import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import 'package:youtube_player_flutter/youtube_player_flutter.dart';
import '../../data/constants/quotes.dart';
import '../../data/models/emotion_model.dart';
import '../../widgets/custom_text.dart';
import '../../widgets/custom_app_bar.dart';
import '../../widgets/emotion_video_player.dart';

class QuotesDetailScreen extends StatelessWidget {
  final EmotionModel emotion;

  const QuotesDetailScreen({super.key, required this.emotion});

  @override
  Widget build(BuildContext context) {
    final quotes = Quotes.getQuotesFor(emotion.type);
    final videos = Quotes.getVideosFor(emotion.type);
    final color = _getEmotionColor(emotion.type);

    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      appBar: CustomAppBar(
        title: "${emotion.type.capitalizeFirst} Reflections",
        showBackButton: true,
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.fromLTRB(16.w, 16.h, 16.w, 32.h),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Quotes Section
            _buildSectionHeader(
              context,
              "Quotes to Ponder",
              Icons.format_quote,
              color,
            ),
            SizedBox(height: 12.h),
            ...quotes.map((quote) => _buildQuoteCard(context, quote, color)),

            SizedBox(height: 32.h),

            // Videos Section
            if (videos.isNotEmpty) ...[
              _buildSectionHeader(
                context,
                "Suggested Videos",
                Icons.play_circle_fill,
                color,
              ),
              SizedBox(height: 12.h),
              SizedBox(
                height: 150.h,
                child: ListView.separated(
                  scrollDirection: Axis.horizontal,
                  itemCount: videos.length,
                  separatorBuilder: (context, index) => SizedBox(width: 12.w),
                  itemBuilder: (context, index) =>
                      _buildVideoCard(context, videos[index], color),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildSectionHeader(
    BuildContext context,
    String title,
    IconData icon,
    Color color,
  ) {
    return Row(
      children: [
        Icon(icon, color: color, size: 24.sp),
        SizedBox(width: 8.w),
        CustomText(
          title,
          fontSize: 18,
          fontWeight: FontWeight.bold,
          color: Theme.of(context).textTheme.bodyLarge?.color,
        ),
      ],
    );
  }

  Widget _buildQuoteCard(BuildContext context, String quote, Color color) {
    return Container(
      margin: EdgeInsets.only(bottom: 12.h),
      padding: EdgeInsets.all(16.w),
      decoration: BoxDecoration(
        color: Theme.of(context).cardColor,
        borderRadius: BorderRadius.circular(12.r),
        border: Border.all(color: color.withValues(alpha: 0.2)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: CustomText(
        "\"$quote\"",
        fontSize: 15,
        fontStyle: FontStyle.italic,
        color: Theme.of(context).textTheme.bodyMedium?.color,
        height: 1.5,
      ),
    );
  }

  Widget _buildVideoCard(
    BuildContext context,
    Map<String, String> video,
    Color color,
  ) {
    final url = video['url'] ?? '';
    final videoId = _extractVideoId(url);
    final hasThumbnail = videoId.isNotEmpty;

    return GestureDetector(
      onTap: () => _playVideoInApp(context, url, video['title'] ?? ''),
      child: Container(
        width: 200.w,
        decoration: BoxDecoration(
          color: Theme.of(context).cardColor,
          borderRadius: BorderRadius.circular(12.r),
          border: Border.all(color: color.withValues(alpha: 0.2)),
        ),
        clipBehavior: Clip.antiAlias,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Thumbnail area
            Expanded(
              child: Stack(
                fit: StackFit.expand,
                children: [
                  // YouTube thumbnail or fallback
                  if (hasThumbnail)
                    Image.network(
                      'https://img.youtube.com/vi/$videoId/hqdefault.jpg',
                      fit: BoxFit.cover,
                      loadingBuilder: (context, child, loadingProgress) {
                        if (loadingProgress == null) return child;
                        return Container(
                          color: color.withValues(alpha: 0.1),
                          child: Center(
                            child: SizedBox(
                              width: 24.w,
                              height: 24.w,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                color: color,
                              ),
                            ),
                          ),
                        );
                      },
                      errorBuilder: (context, error, stackTrace) {
                        // Network error or thumbnail unavailable — icon fallback
                        return Container(
                          color: color.withValues(alpha: 0.1),
                          child: Center(
                            child: Icon(
                              Icons.play_circle_outline,
                              color: color,
                              size: 40.sp,
                            ),
                          ),
                        );
                      },
                    )
                  else
                    // No valid video ID — icon fallback
                    Container(
                      color: color.withValues(alpha: 0.1),
                      child: Center(
                        child: Icon(
                          Icons.play_circle_outline,
                          color: color,
                          size: 40.sp,
                        ),
                      ),
                    ),

                  // Play button overlay on top of the thumbnail
                  if (hasThumbnail)
                    Center(
                      child: Container(
                        padding: EdgeInsets.all(8.w),
                        decoration: BoxDecoration(
                          color: Colors.black.withValues(alpha: 0.45),
                          shape: BoxShape.circle,
                        ),
                        child: Icon(
                          Icons.play_arrow_rounded,
                          color: Colors.white,
                          size: 28.sp,
                        ),
                      ),
                    ),
                ],
              ),
            ),
            // Title
            Padding(
              padding: EdgeInsets.all(12.w),
              child: CustomText(
                video['title'] ?? '',
                fontSize: 14,
                fontWeight: FontWeight.bold,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// Extracts YouTube video ID from any standard YouTube URL format.
  /// Returns an empty string if extraction fails (malformed / non-YouTube URL).
  String _extractVideoId(String url) {
    if (url.isEmpty) return '';
    return YoutubePlayer.convertUrlToId(url) ?? '';
  }

  void _playVideoInApp(BuildContext context, String url, String title) {
    if (url.isEmpty) return;

    Get.dialog(
      Dialog(
        backgroundColor: Colors.transparent,
        insetPadding: EdgeInsets.all(16.w),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              decoration: BoxDecoration(
                color: Theme.of(context).scaffoldBackgroundColor,
                borderRadius: BorderRadius.circular(16.r),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // Header
                  Padding(
                    padding: EdgeInsets.all(16.w),
                    child: Row(
                      children: [
                        Expanded(
                          child: CustomText(
                            title,
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                        IconButton(
                          icon: const Icon(Icons.close),
                          onPressed: () => Get.back(),
                          visualDensity: VisualDensity.compact,
                        ),
                      ],
                    ),
                  ),
                  // Player
                  ClipRRect(
                    borderRadius: BorderRadius.only(
                      bottomLeft: Radius.circular(16.r),
                      bottomRight: Radius.circular(16.r),
                    ),
                    child: EmotionVideoPlayer(
                      emotionType: '', // Not used when videoUrl is provided
                      videoUrl: url,
                      height: 220.h,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
      barrierDismissible: true,
    );
  }

  Color _getEmotionColor(String type) {
    switch (type.toLowerCase()) {
      case 'happy':
        return const Color(0xFFF39C12); // AppColors.emotionHappy
      case 'sad':
        return const Color(0xFF5D6D7E); // AppColors.emotionSad
      case 'angry':
        return const Color(0xFFE74C3C); // AppColors.emotionAngry
      case 'fear':
        return const Color(0xFF8E44AD); // AppColors.emotionFear
      case 'surprise':
        return const Color(0xFF3498DB); // AppColors.emotionSurprise
      case 'disgust':
        return const Color(0xFF8B4513); // AppColors.emotionDisgust
      case 'neutral':
        return const Color(0xFF95A5A6); // AppColors.emotionNeutral
      default:
        return const Color(0xFF4ECDC4); // AppColors.primary
    }
  }
}
