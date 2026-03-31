import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import '../data/constants/colors.dart';

/// Camera button with breathing animation
class BreathingCameraButton extends StatefulWidget {
  final VoidCallback onTap;
  final bool isSelected;

  const BreathingCameraButton({
    super.key,
    required this.onTap,
    this.isSelected = false,
  });

  @override
  State<BreathingCameraButton> createState() => _BreathingCameraButtonState();
}

class _BreathingCameraButtonState extends State<BreathingCameraButton>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _breathingAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(seconds: 2),
      vsync: this,
    );

    _breathingAnimation = Tween<double>(
      begin: 1.0,
      end: 1.15,
    ).animate(CurvedAnimation(parent: _controller, curve: Curves.easeInOut));

    _controller.repeat(reverse: true);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: widget.onTap,
      child: AnimatedBuilder(
        animation: _breathingAnimation,
        builder: (context, child) {
          return Transform.scale(
            scale: _breathingAnimation.value,
            child: Container(
              width: 56.w,
              height: 56.h,
              decoration: BoxDecoration(
                color: AppColors.primary,
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(
                    color: AppColors.primary.withValues(
                      alpha: 0.4 * _breathingAnimation.value,
                    ),
                    blurRadius: 12 * _breathingAnimation.value,
                    spreadRadius: 2 * _breathingAnimation.value,
                    offset: const Offset(0, 4),
                  ),
                ],
              ),
              child: Icon(Icons.camera_alt, color: Colors.white, size: 28.sp),
            ),
          );
        },
      ),
    );
  }
}
