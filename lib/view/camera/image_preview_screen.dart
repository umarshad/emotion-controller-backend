import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import '../../data/models/emotion_model.dart';
import '../../data/services/emotion_api_service.dart';
import '../../widgets/custom_text.dart';
import '../../data/controllers/emotion_controller.dart';
import '../../data/controllers/navigation_controller.dart';
import '../history/widgets/detailed_emotion_view.dart';
import '../../widgets/emotion_rating_dialog.dart';

class ImagePreviewScreen extends StatefulWidget {
  final String imagePath;
  final EmotionModel? emotion;

  const ImagePreviewScreen({super.key, required this.imagePath, this.emotion});

  @override
  State<ImagePreviewScreen> createState() => _ImagePreviewScreenState();
}

class _ImagePreviewScreenState extends State<ImagePreviewScreen> {
  EmotionModel? _displayedEmotion;
  bool _isAnalyzing = false;

  @override
  void initState() {
    super.initState();
    _displayedEmotion = widget.emotion;
    if (_displayedEmotion == null) {
      _analyzeImage();
    }
  }

  Future<void> _analyzeImage() async {
    setState(() {
      _isAnalyzing = true;
    });

    try {
      final file = File(widget.imagePath);
      if (await file.exists()) {
        final bytes = await file.readAsBytes();
        // Better:
        // The Service expects Uint8List.
        final result = await EmotionApiService().detectEmotionFromCamera(
          imageBytes: bytes,
          skipCompression: false,
        );

        if (mounted) {
          setState(() {
            _displayedEmotion = result;
          });
        }
      }
    } catch (e) {
      debugPrint('[ImagePreview] Analysis failed: $e');
    } finally {
      if (mounted) {
        setState(() {
          _isAnalyzing = false;
        });
      }
    }
  }

  void _viewDetails() {
    if (_displayedEmotion == null) {
      if (_isAnalyzing) {
        Get.snackbar("Wait", "Please wait for analysis to complete");
      } else {
        Get.snackbar("Error", "No emotion detected. Please retake.");
      }
      return;
    }

    // Navigate to Detailed View with onSave callback
    Get.to(
      () => DetailedEmotionView(
        emotion: _displayedEmotion!,
        onSave: _saveAndExit,
      ),
    );
  }

  Future<void> _saveAndExit() async {
    try {
      // 1. Update emotion with image path
      final emotionToSave = _displayedEmotion!.copyWith(
        imagePath: widget.imagePath,
      );

      // 2. Save using controller
      final emotionController = Get.find<EmotionController>();
      emotionController.addEmotion(emotionToSave);

      // 3. Navigate to History or Home
      Get.snackbar(
        "Saved",
        "Emotion saved to history",
        backgroundColor: Colors.green.withValues(alpha: 0.8),
        colorText: Colors.white,
      );

      // 3. Navigate to Home and show rating dialog
      Get.until((route) => route.isFirst); // Pop to root
      
      try {
        final navController = Get.find<NavigationController>();
        navController.changeIndex(0); // Go to Home
      } catch (e) {
        // Fallback
      }

      // Show Rating Dialog
      Get.dialog(
        EmotionRatingDialog(
          emotionId: emotionToSave.id,
          onRatingSelected: (rating) {
            emotionController.updateEmotionRating(emotionToSave.id, rating);
          },
        ),
        barrierDismissible: false,
      );
    } catch (e) {
      debugPrint('[ImagePreview] Save Error: $e');
      Get.snackbar("Error", "Failed to save emotion");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // 1. Image
          Image.file(File(widget.imagePath), fit: BoxFit.cover),

          // 2. Overlay Gradient
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            height: 200.h,
            child: Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    Colors.transparent,
                    Colors.black.withValues(alpha: 0.8),
                  ],
                ),
              ),
            ),
          ),

          // 3. Emotion Result Card
          Positioned(
            bottom: 40.h,
            left: 20.w,
            right: 20.w,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              mainAxisSize: MainAxisSize.min,
              children: [
                if (_isAnalyzing)
                  Container(
                    padding: EdgeInsets.all(16.w),
                    decoration: BoxDecoration(
                      color: Colors.black54,
                      borderRadius: BorderRadius.circular(12.r),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        SizedBox(
                          width: 20.w,
                          height: 20.w,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Colors.white,
                          ),
                        ),
                        SizedBox(width: 12.w),
                        CustomText(
                          "Analyzing...",
                          color: Colors.white,
                          fontSize: 16,
                        ),
                      ],
                    ),
                  )
                else if (_displayedEmotion != null) ...[
                  Container(
                    padding: EdgeInsets.symmetric(
                      horizontal: 20.w,
                      vertical: 10.h,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.white.withValues(alpha: 0.15),
                      borderRadius: BorderRadius.circular(20.r),
                      border: Border.all(
                        color: Colors.white.withValues(alpha: 0.3),
                        width: 1,
                      ),
                    ),
                    child: Column(
                      children: [
                        CustomText(
                          _displayedEmotion!.type.toUpperCase(),
                          color: _getEmotionColor(_displayedEmotion!.type),
                          fontSize: 28,
                          fontWeight: FontWeight.bold,
                          letterSpacing: 1.2,
                        ),
                        SizedBox(height: 4.h),
                        CustomText(
                          "Confidence: ${_displayedEmotion!.intensity.toStringAsFixed(1)}%",
                          color: Colors.white70,
                          fontSize: 14,
                        ),
                      ],
                    ),
                  ),
                ],

                SizedBox(height: 30.h),

                // Action Buttons
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    // Retake
                    _buildActionButton(
                      icon: Icons.close,
                      label: "Retake",
                      onTap: () => Get.back(),
                      color: Colors.white,
                    ),
                    // View Details
                    _buildActionButton(
                      icon: Icons.arrow_forward,
                      label: "Details",
                      onTap: _viewDetails,
                      color: Colors.white,
                      isPrimary: true,
                    ),
                  ],
                ),
              ],
            ),
          ),

          // Close button top left
          Positioned(
            top: 40.h,
            left: 20.w,
            child: IconButton(
              icon: Icon(Icons.close, color: Colors.white, size: 28.sp),
              onPressed: () => Get.back(),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildActionButton({
    required IconData icon,
    required String label,
    required VoidCallback onTap,
    required Color color,
    bool isPrimary = false,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            padding: EdgeInsets.all(16.w),
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: isPrimary ? color : Colors.white.withValues(alpha: 0.1),
              border: isPrimary ? null : Border.all(color: Colors.white54),
            ),
            child: Icon(
              icon,
              color: isPrimary ? Colors.black : Colors.white,
              size: 28.sp,
            ),
          ),
          SizedBox(height: 8.h),
          CustomText(
            label,
            color: Colors.white,
            fontSize: 14,
            fontWeight: FontWeight.w500,
          ),
        ],
      ),
    );
  }

  Color _getEmotionColor(String type) {
    switch (type.toLowerCase()) {
      case 'happy':
        return const Color(0xFFFFD700);
      case 'sad':
        return const Color(0xFF2196F3);
      case 'angry':
        return const Color(0xFFF44336);
      case 'surprise':
        return const Color(0xFFFF9800);
      case 'fear':
        return const Color(0xFF9C27B0);
      default:
        return const Color(0xFF4CAF50);
    }
  }
}
