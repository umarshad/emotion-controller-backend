import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:youtube_player_flutter/youtube_player_flutter.dart';
import 'package:url_launcher/url_launcher.dart';
import '../data/constants/colors.dart';
import '../data/constants/video_urls.dart';
import 'custom_text.dart';

/// Reusable YouTube video player widget for emotions
class EmotionVideoPlayer extends StatefulWidget {
  final String emotionType;
  final String? videoUrl; // New optional parameter
  final double? height;

  const EmotionVideoPlayer({
    super.key,
    required this.emotionType,
    this.videoUrl,
    this.height,
  });

  @override
  State<EmotionVideoPlayer> createState() => _EmotionVideoPlayerState();
}

class _EmotionVideoPlayerState extends State<EmotionVideoPlayer> {
  YoutubePlayerController? _controller;
  bool _isInitialized = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _initializePlayer();
  }

  @override
  void didUpdateWidget(EmotionVideoPlayer oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.emotionType != widget.emotionType ||
        oldWidget.videoUrl != widget.videoUrl) {
      _controller?.dispose();
      _initializePlayer();
    }
  }

  void _initializePlayer() {
    try {
      // Priority: Specific URL (if not empty) > Emotion Default
      String urlToUse;
      if (widget.videoUrl != null && widget.videoUrl!.isNotEmpty) {
        urlToUse = widget.videoUrl!;
      } else {
        urlToUse = EmotionVideoUrls.getVideoUrl(widget.emotionType);
      }

      final videoId = EmotionVideoUrls.extractVideoId(urlToUse);

      if (videoId.isEmpty) {
        setState(() {
          _error = 'Invalid video URL';
        });
        return;
      }

      _controller = YoutubePlayerController(
        initialVideoId: videoId,
        flags: const YoutubePlayerFlags(
          autoPlay: false,
          mute: false,
          enableCaption: true,
          controlsVisibleAtStart: true,
          hideControls: false,
          loop: false,
          isLive: false,
        ),
      );

      _controller!.addListener(_controllerListener);

      setState(() {
        _isInitialized = true;
      });
    } catch (e) {
      setState(() {
        _error = 'Failed to load video: $e';
        _isInitialized = false;
      });
    }
  }

  void _controllerListener() {
    if (!mounted) return;

    try {
      if (_controller?.value.hasError == true) {
        final code = _controller?.value.errorCode;
        setState(() {
          _error = 'Video player error: ${code ?? "Unknown error"}';
          _isInitialized = false;
        });
      } else if (_controller?.value.isReady == true) {
        setState(() {
          _isInitialized = true;
          _error = null;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = 'Error handling video: $e';
          _isInitialized = false;
        });
      }
    }
  }

  Future<void> _launchVideoUrl() async {
    if (_controller == null) return;
    final url =
        'https://www.youtube.com/watch?v=${_controller!.initialVideoId}';
    final uri = Uri.parse(url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }

  @override
  void dispose() {
    _controller?.removeListener(_controllerListener);
    _controller?.dispose();
    super.dispose();
  }

  Widget _buildFallbackUI(BuildContext context) {
    return Container(
      margin: EdgeInsets.symmetric(vertical: 8.h),
      decoration: BoxDecoration(
        color: AppColors.cardBackground,
        borderRadius: BorderRadius.circular(12.r),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: EdgeInsets.all(16.w),
            child: Row(
              children: [
                Icon(
                  Icons.play_circle_outline,
                  color: AppColors.primary,
                  size: 24.sp,
                ),
                SizedBox(width: 8.w),
                CustomText(
                  'Helpful Video',
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ],
            ),
          ),
          Stack(
            children: [
              AspectRatio(
                aspectRatio: 16 / 9,
                child: Image.network(
                  'https://img.youtube.com/vi/${_controller?.initialVideoId ?? ""}/hqdefault.jpg',
                  fit: BoxFit.cover,
                  errorBuilder: (context, error, stackTrace) => Container(
                    color: Colors.black12,
                    child: Icon(Icons.broken_image, color: AppColors.textLight),
                  ),
                ),
              ),
              Positioned.fill(
                child: Container(
                  color: Colors.black.withValues(alpha: 0.7),
                  child: Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.error_outline,
                          color: Colors.white,
                          size: 32.sp,
                        ),
                        SizedBox(height: 8.h),
                        CustomText(
                          'Playback Restricted',
                          fontSize: 16,
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                        ),
                        SizedBox(height: 4.h),
                        CustomText(
                          'This video cannot be played in the app.',
                          fontSize: 12,
                          color: Colors.white70,
                        ),
                        SizedBox(height: 16.h),
                        ElevatedButton.icon(
                          onPressed: _launchVideoUrl,
                          icon: Icon(Icons.open_in_new, size: 16.sp),
                          label: Text("Watch on YouTube"),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: AppColors.primary,
                            foregroundColor: Colors.white,
                            padding: EdgeInsets.symmetric(
                              horizontal: 16.w,
                              vertical: 8.h,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    // 1. Error State
    if (_error != null) {
      if (_error!.contains('150') ||
          _error!.contains('101') ||
          _error!.contains('100')) {
        return _buildFallbackUI(context);
      }

      return _buildContainer(
        context,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.error_outline, color: AppColors.error, size: 32.sp),
            SizedBox(height: 8.h),
            CustomText(
              _error!,
              fontSize: 14,
              color: AppColors.textSecondary,
              textAlign: TextAlign.center,
            ),
          ],
        ),
      );
    }

    // 2. Loading State
    if (!_isInitialized || _controller == null) {
      return _buildContainer(
        context,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(color: AppColors.primary),
            SizedBox(height: 16.h),
            CustomText(
              'Loading video preview...',
              fontSize: 14,
              color: AppColors.textSecondary,
            ),
          ],
        ),
      );
    }

    return Container(
      margin: EdgeInsets.symmetric(vertical: 8.h),
      decoration: BoxDecoration(
        color: AppColors.cardBackground,
        borderRadius: BorderRadius.circular(12.r),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: EdgeInsets.all(16.w),
            child: Row(
              children: [
                Icon(
                  Icons.play_circle_outline,
                  color: AppColors.primary,
                  size: 24.sp,
                ),
                SizedBox(width: 8.w),
                CustomText(
                  'Helpful Video',
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ],
            ),
          ),
          Builder(
            builder: (context) {
              try {
                return YoutubePlayer(
                  controller: _controller!,
                  showVideoProgressIndicator: true,
                  progressIndicatorColor: AppColors.primary,
                  thumbnail: Image.network(
                    'https://img.youtube.com/vi/${_controller!.initialVideoId}/hqdefault.jpg',
                    fit: BoxFit.cover,
                    errorBuilder: (context, error, stackTrace) {
                      return Container(
                        color: Colors.black12,
                        child: Icon(
                          Icons.broken_image,
                          color: AppColors.textLight,
                        ),
                      );
                    },
                  ),
                  progressColors: ProgressBarColors(
                    playedColor: AppColors.primary,
                    handleColor: AppColors.primary,
                    backgroundColor: AppColors.border,
                    bufferedColor: AppColors.textLight,
                  ),
                  onReady: () {
                    if (mounted) {
                      setState(() {
                        _isInitialized = true;
                        _error = null;
                      });
                    }
                  },
                  onEnded: (metadata) {
                    // Video ended
                  },
                );
              } catch (e) {
                // Return fallback if YoutubePlayer widget itself throws an error
                return Container(
                  height: 200.h,
                  padding: EdgeInsets.all(16.w),
                  child: Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.error_outline,
                          color: AppColors.error,
                          size: 32.sp,
                        ),
                        SizedBox(height: 8.h),
                        CustomText(
                          'Unable to display video player',
                          fontSize: 14,
                          color: AppColors.textSecondary,
                          textAlign: TextAlign.center,
                        ),
                      ],
                    ),
                  ),
                );
              }
            },
          ),
        ],
      ),
    );
  }

  Widget _buildContainer(BuildContext context, {required Widget child}) {
    return Container(
      width: double.infinity,
      padding: EdgeInsets.all(24.w),
      margin: EdgeInsets.symmetric(vertical: 8.h),
      decoration: BoxDecoration(
        color: AppColors.cardBackground,
        borderRadius: BorderRadius.circular(12.r),
        border: Border.all(color: AppColors.border),
      ),
      child: Center(child: child),
    );
  }
}
