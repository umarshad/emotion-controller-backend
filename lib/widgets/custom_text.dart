import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import '../data/constants/colors.dart';

/// Custom responsive text widget using ScreenUtil
class CustomText extends StatelessWidget {
  final String text;
  final double? fontSize;
  final FontWeight? fontWeight;
  final Color? color;
  final TextAlign? textAlign;
  final int? maxLines;
  final TextOverflow? overflow;
  final double? letterSpacing;
  final double? height;
  final FontStyle? fontStyle;

  const CustomText(
    this.text, {
    super.key,
    this.fontSize,
    this.fontWeight,
    this.color,
    this.textAlign,
    this.maxLines,
    this.overflow,
    this.letterSpacing,
    this.height,
    this.fontStyle,
  });

  @override
  Widget build(BuildContext context) {
    return Text(
      text,
      style: TextStyle(
        fontSize: fontSize?.sp ?? 14.sp,
        fontWeight: fontWeight ?? FontWeight.normal,
        color: color ?? AppColors.textPrimary,
        letterSpacing: letterSpacing,
        height: height,
        fontStyle: fontStyle,
      ),
      textAlign: textAlign,
      maxLines: maxLines,
      overflow: overflow ?? TextOverflow.visible,
    );
  }
}

/// Heading text variants
class HeadingText extends CustomText {
  const HeadingText(
    super.text, {
    super.key,
    super.color,
    super.textAlign,
    super.maxLines,
  }) : super(fontSize: 24, fontWeight: FontWeight.bold);
}

class SubHeadingText extends CustomText {
  const SubHeadingText(
    super.text, {
    super.key,
    super.color,
    super.textAlign,
    super.maxLines,
  }) : super(fontSize: 18, fontWeight: FontWeight.w600);
}

class BodyText extends CustomText {
  const BodyText(
    super.text, {
    super.key,
    super.color,
    super.textAlign,
    super.maxLines,
  }) : super(fontSize: 14, fontWeight: FontWeight.normal);
}

class CaptionText extends CustomText {
  const CaptionText(
    super.text, {
    super.key,
    super.color,
    super.textAlign,
    super.maxLines,
  }) : super(fontSize: 12, fontWeight: FontWeight.normal);
}
