import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import '../data/constants/colors.dart';
import 'custom_text.dart';

class EmotionRatingDialog extends StatefulWidget {
  final String emotionId;
  final int initialRating;
  final Function(int) onRatingSelected;

  const EmotionRatingDialog({
    super.key,
    required this.emotionId,
    this.initialRating = 0,
    required this.onRatingSelected,
  });

  @override
  State<EmotionRatingDialog> createState() => _EmotionRatingDialogState();
}

class _EmotionRatingDialogState extends State<EmotionRatingDialog> {
  late int _currentRating;

  @override
  void initState() {
    super.initState();
    _currentRating = widget.initialRating;
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20.r)),
      elevation: 0,
      backgroundColor: Colors.transparent,
      child: _contentBox(context),
    );
  }

  Widget _contentBox(BuildContext context) {
    return Container(
      padding: EdgeInsets.all(24.w),
      decoration: BoxDecoration(
        shape: BoxShape.rectangle,
        color: Theme.of(context).cardColor,
        borderRadius: BorderRadius.circular(20.r),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.1),
            offset: const Offset(0, 10),
            blurRadius: 10,
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            padding: EdgeInsets.all(16.w),
            decoration: BoxDecoration(
              color: AppColors.primary.withValues(alpha: 0.1),
              shape: BoxShape.circle,
            ),
            child: Icon(
              Icons.stars_rounded,
              color: AppColors.primary,
              size: 40.sp,
            ),
          ),
          SizedBox(height: 20.h),
          CustomText(
            "How accurate was this?",
            fontSize: 20,
            fontWeight: FontWeight.bold,
            textAlign: TextAlign.center,
          ),
          SizedBox(height: 12.h),
          CustomText(
            "Your feedback helps us improve the AI detection for better results.",
            fontSize: 14,
            color: AppColors.textSecondary,
            textAlign: TextAlign.center,
          ),
          SizedBox(height: 24.h),
          LayoutBuilder(
            builder: (context, constraints) {
              // Calculate available space and optimal star size
              final double maxStarSize = 40.sp;
              final double availableWidth = constraints.maxWidth;
              // 5 stars + some padding/gap
              final double starSize = (availableWidth / 5.5).clamp(24.0, maxStarSize);

              return Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: List.generate(5, (index) {
                  return Flexible(
                    child: TweenAnimationBuilder<double>(
                      tween: Tween(begin: 0.0, end: 1.0),
                      duration: Duration(milliseconds: 200 + (index * 100)),
                      builder: (context, value, child) {
                        return Transform.scale(
                          scale: value,
                          child: IconButton(
                            padding: EdgeInsets.zero,
                            constraints: BoxConstraints(
                              minWidth: starSize,
                              minHeight: starSize,
                            ),
                            icon: Icon(
                              index < _currentRating
                                  ? Icons.star_rounded
                                  : Icons.star_outline_rounded,
                              color: index < _currentRating
                                  ? Colors.amber
                                  : AppColors.textLight.withValues(alpha: 0.5),
                              size: starSize,
                            ),
                            onPressed: () {
                              setState(() {
                                _currentRating = index + 1;
                              });
                            },
                          ),
                        );
                      },
                    ),
                  );
                }),
              );
            },
          ),
          SizedBox(height: 32.h),
          Row(
            children: [
              Expanded(
                child: TextButton(
                  onPressed: () => Get.back(),
                  style: TextButton.styleFrom(
                    padding: EdgeInsets.symmetric(vertical: 12.h),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12.r),
                    ),
                  ),
                  child: CustomText(
                    "Skip",
                    color: AppColors.textSecondary,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              SizedBox(width: 16.w),
              Expanded(
                child: ElevatedButton(
                  onPressed: _currentRating > 0
                      ? () {
                          widget.onRatingSelected(_currentRating);
                          Get.back();
                          _showSuccessSnackBar();
                        }
                      : null,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.primary,
                    foregroundColor: Colors.white,
                    padding: EdgeInsets.symmetric(vertical: 12.h),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12.r),
                    ),
                    elevation: 0,
                  ),
                  child: const Text("Rate Now"),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  void _showSuccessSnackBar() {
    Get.snackbar(
      "Thank you!",
      "We've received your feedback.",
      snackPosition: SnackPosition.BOTTOM,
      backgroundColor: AppColors.primary.withValues(alpha: 0.9),
      colorText: Colors.white,
      margin: EdgeInsets.all(16.w),
      borderRadius: 12.r,
      duration: const Duration(seconds: 2),
    );
  }
}
