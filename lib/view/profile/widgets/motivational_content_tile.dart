import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import 'dart:math';
import '../../../data/constants/colors.dart';
import '../../../widgets/custom_text.dart';
import 'motivational_bottom_sheet.dart';

/// Dynamic motivational content widget with rotating quotes and tips
class MotivationalContentTile extends StatefulWidget {
  const MotivationalContentTile({super.key});

  @override
  State<MotivationalContentTile> createState() =>
      _MotivationalContentTileState();
}

class _MotivationalContentTileState extends State<MotivationalContentTile>
    with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;
  int _currentIndex = 0;
  final Random _random = Random();

  // List of motivational quotes and tips
  final List<String> _motivationalContent = [
    "Your emotions are valid. Take time to understand them.",
    "Self-awareness is the first step to emotional well-being.",
    "It's okay to not be okay. Reach out for support when needed.",
    "Mindfulness can help you manage difficult emotions.",
    "Every emotion serves a purpose. Listen to what yours are telling you.",
    "Taking care of your emotional health is just as important as physical health.",
    "You don't have to face your emotions alone. Support is available.",
    "Recognizing your feelings is a sign of strength, not weakness.",
    "Emotional well-being is a journey, not a destination.",
    "Give yourself permission to feel and process your emotions.",
    "Small steps toward emotional awareness can lead to big changes.",
    "Your emotional state matters. Take time to check in with yourself.",
  ];

  @override
  void initState() {
    super.initState();
    _currentIndex = _random.nextInt(_motivationalContent.length);

    _animationController = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );

    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeInOut),
    );

    _animationController.forward();
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  void _showBottomSheet(BuildContext context) {
    Get.bottomSheet(
      MotivationalBottomSheet(quotes: _motivationalContent),
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
    );
  }

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: () => _showBottomSheet(context),
      child: Container(
        padding: EdgeInsets.symmetric(horizontal: 16.w, vertical: 20.h),
        decoration: BoxDecoration(
          color: Theme.of(context).cardColor,
          borderRadius: BorderRadius.circular(16.r),
          border: Border.all(color: Theme.of(context).dividerColor),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 40.w,
              height: 40.h,
              decoration: BoxDecoration(
                color: AppColors.primary.withValues(alpha: 0.1),
                shape: BoxShape.circle,
              ),
              child: Icon(
                Icons.lightbulb_outline,
                color: AppColors.primary,
                size: 24.sp,
              ),
            ),
            SizedBox(width: 16.w),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  CustomText(
                    'Daily Inspiration',
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color:
                        Theme.of(context).textTheme.bodyMedium?.color ??
                        AppColors.textSecondary,
                  ),
                  SizedBox(height: 8.h),
                  FadeTransition(
                    opacity: _fadeAnimation,
                    child: CustomText(
                      _motivationalContent[_currentIndex],
                      fontSize: 16,
                      fontWeight: FontWeight.w500,
                      color:
                          Theme.of(context).textTheme.bodyLarge?.color ??
                          AppColors.textPrimary,
                    ),
                  ),
                  SizedBox(height: 8.h),
                  Text(
                    'Tap to see another',
                    style: TextStyle(
                      fontSize: 12.sp,
                      color:
                          Theme.of(context).textTheme.bodySmall?.color ??
                          AppColors.textLight,
                      fontStyle: FontStyle.italic,
                    ),
                  ),
                ],
              ),
            ),
            Icon(
              Icons.arrow_forward_ios,
              size: 16.sp,
              color:
                  Theme.of(context).textTheme.bodySmall?.color ??
                  AppColors.textLight,
            ),
          ],
        ),
      ),
    );
  }
}
