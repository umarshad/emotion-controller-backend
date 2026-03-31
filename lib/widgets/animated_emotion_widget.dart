import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import '../data/constants/colors.dart';
import '../data/constants/assets.dart';
import '../data/utils/animations.dart';

/// Animated visual representation of emotions (breathing, pulsing effects)
class AnimatedEmotionWidget extends StatefulWidget {
  final String emotionType;
  final double size;
  final bool showAnimation;

  const AnimatedEmotionWidget({
    super.key,
    required this.emotionType,
    this.size = 100,
    this.showAnimation = true,
  });

  @override
  State<AnimatedEmotionWidget> createState() => _AnimatedEmotionWidgetState();
}

class _AnimatedEmotionWidgetState extends State<AnimatedEmotionWidget>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: AppAnimations.getEmotionDuration(widget.emotionType),
      vsync: this,
    );

    final curve = AppAnimations.getEmotionCurve(widget.emotionType);
    _animation = Tween<double>(
      begin: 0.8,
      end: 1.2,
    ).animate(CurvedAnimation(parent: _controller, curve: curve));

    if (widget.showAnimation) {
      _controller.repeat(reverse: true);
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final emotionColor = getEmotionColor(widget.emotionType);
    final emoji = AppAssets.getEmotionEmoji(widget.emotionType);

    return AnimatedBuilder(
      animation: _animation,
      builder: (context, child) {
        return Container(
          width: widget.size.w,
          height: widget.size.h,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: emotionColor.withValues(alpha: 0.1),
            boxShadow: [
              BoxShadow(
                color: emotionColor.withValues(alpha: 0.3 * _animation.value),
                blurRadius: 20 * _animation.value,
                spreadRadius: 5 * _animation.value,
              ),
            ],
          ),
          child: Center(
            child: Transform.scale(
              scale: _animation.value,
              child: Text(
                emoji,
                style: TextStyle(fontSize: (widget.size * 0.5).sp),
              ),
            ),
          ),
        );
      },
    );
  }
}

/// Breathing animation for calm/relaxed emotions
class BreathingEmotionWidget extends StatefulWidget {
  final String emotionType;
  final double size;

  const BreathingEmotionWidget({
    super.key,
    required this.emotionType,
    this.size = 100,
  });

  @override
  State<BreathingEmotionWidget> createState() => _BreathingEmotionWidgetState();
}

class _BreathingEmotionWidgetState extends State<BreathingEmotionWidget>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _breathingAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: AppAnimations.breathingDuration,
      vsync: this,
    );

    _breathingAnimation = Tween<double>(
      begin: 0.9,
      end: 1.1,
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
    final emotionColor = getEmotionColor(widget.emotionType);
    final emoji = AppAssets.getEmotionEmoji(widget.emotionType);

    return AnimatedBuilder(
      animation: _breathingAnimation,
      builder: (context, child) {
        return Container(
          width: widget.size.w,
          height: widget.size.h,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: emotionColor.withValues(
              alpha: 0.15 * _breathingAnimation.value,
            ),
          ),
          child: Center(
            child: Transform.scale(
              scale: _breathingAnimation.value,
              child: Text(
                emoji,
                style: TextStyle(fontSize: (widget.size * 0.5).sp),
              ),
            ),
          ),
        );
      },
    );
  }
}

/// Pulsing animation for stress/anxiety
class PulsingEmotionWidget extends StatefulWidget {
  final String emotionType;
  final double size;

  const PulsingEmotionWidget({
    super.key,
    required this.emotionType,
    this.size = 100,
  });

  @override
  State<PulsingEmotionWidget> createState() => _PulsingEmotionWidgetState();
}

class _PulsingEmotionWidgetState extends State<PulsingEmotionWidget>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: AppAnimations.pulsingDuration,
      vsync: this,
    );

    _pulseAnimation = Tween<double>(
      begin: 1.0,
      end: 1.3,
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
    final emotionColor = getEmotionColor(widget.emotionType);
    final emoji = AppAssets.getEmotionEmoji(widget.emotionType);

    return AnimatedBuilder(
      animation: _pulseAnimation,
      builder: (context, child) {
        return Container(
          width: widget.size.w,
          height: widget.size.h,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: emotionColor.withValues(alpha: 0.2),
            border: Border.all(
              color: emotionColor,
              width: 2 * _pulseAnimation.value,
            ),
          ),
          child: Center(
            child: Transform.scale(
              scale: _pulseAnimation.value,
              child: Text(
                emoji,
                style: TextStyle(fontSize: (widget.size * 0.5).sp),
              ),
            ),
          ),
        );
      },
    );
  }
}
