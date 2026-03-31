import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import '../data/constants/colors.dart';

/// Get emotion color by emotion type
/// Uses the 7 backend emotions
Color getEmotionColor(String emotionType) {
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

/// Custom button with emotion-based styling
class CustomButton extends StatelessWidget {
  final String text;
  final VoidCallback? onPressed;
  final Color? backgroundColor;
  final Color? textColor;
  final double? width;
  final double? height;
  final double? borderRadius;
  final EdgeInsets? padding;
  final Widget? icon;
  final bool isLoading;
  final bool isOutlined;

  const CustomButton({
    super.key,
    required this.text,
    this.onPressed,
    this.backgroundColor,
    this.textColor,
    this.width,
    this.height,
    this.borderRadius,
    this.padding,
    this.icon,
    this.isLoading = false,
    this.isOutlined = false,
  });

  @override
  Widget build(BuildContext context) {
    final buttonStyle = isOutlined
        ? OutlinedButton.styleFrom(
            backgroundColor: Colors.transparent,
            foregroundColor: textColor ?? backgroundColor ?? AppColors.primary,
            side: BorderSide(
              color: backgroundColor ?? AppColors.primary,
              width: 2.w,
            ),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(
                borderRadius ?? 12.r,
              ),
            ),
            padding: padding ?? EdgeInsets.symmetric(horizontal: 24.w, vertical: 14.h),
            minimumSize: Size(width ?? double.infinity, height ?? 50.h),
          )
        : ElevatedButton.styleFrom(
            backgroundColor: backgroundColor ?? AppColors.primary,
            foregroundColor: textColor ?? Colors.white,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(
                borderRadius ?? 12.r,
              ),
            ),
            padding: padding ?? EdgeInsets.symmetric(horizontal: 24.w, vertical: 14.h),
            minimumSize: Size(width ?? double.infinity, height ?? 50.h),
            elevation: 2,
          );

    final button = isOutlined
        ? OutlinedButton(
            onPressed: isLoading ? null : onPressed,
            style: buttonStyle,
            child: _buildButtonContent(),
          )
        : ElevatedButton(
            onPressed: isLoading ? null : onPressed,
            style: buttonStyle,
            child: _buildButtonContent(),
          );

    return SizedBox(
      width: width,
      height: height ?? 50.h,
      child: button,
    );
  }

  Widget _buildButtonContent() {
    if (isLoading) {
      return SizedBox(
        height: 20.h,
        width: 20.w,
        child: CircularProgressIndicator(
          strokeWidth: 2,
          valueColor: AlwaysStoppedAnimation<Color>(
            textColor ?? Colors.white,
          ),
        ),
      );
    }

    if (icon != null) {
      return Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          icon!,
          SizedBox(width: 8.w),
          Text(
            text,
            style: TextStyle(
              fontSize: 16.sp,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      );
    }

    return Text(
      text,
      style: TextStyle(
        fontSize: 16.sp,
        fontWeight: FontWeight.w600,
      ),
    );
  }
}

/// Emotion-based button variant
class EmotionButton extends CustomButton {
  final String emotionType;

  EmotionButton({
    super.key,
    required super.text,
    required this.emotionType,
    super.onPressed,
    super.width,
    super.height,
    super.isLoading,
  }) : super(
          backgroundColor: getEmotionColor(emotionType),
          textColor: Colors.white,
        );
}