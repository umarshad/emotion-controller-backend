import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import '../../data/constants/colors.dart';
import '../../data/constants/strings.dart';
import '../main_navigation.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with SingleTickerProviderStateMixin {
  // List of emojis representing the 7 trained emotions
  final List<String> _emojis = [
    '😠', // Angry
    '🤢', // Disgust
    '😨', // Fear
    '😀', // Happy
    '😐', // Neutral
    '😢', // Sad
    '😮', // Surprise
  ];

  late AnimationController _controller;
  late Animation<double> _scaleAnimation;
  late Animation<double> _fadeAnimation;
  late Animation<double> _logoFadeAnimation;

  int _currentIndex = 0;
  Timer? _emojiTimer;

  @override
  void initState() {
    super.initState();

    // Initialize animation controller for the emoji pulse effect
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 300),
    );

    _scaleAnimation = Tween<double>(
      begin: 0.5,
      end: 1.2,
    ).animate(CurvedAnimation(parent: _controller, curve: Curves.easeOutBack));

    _fadeAnimation = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(parent: _controller, curve: Curves.easeIn));

    _logoFadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
        parent: _controller,
        curve: const Interval(0.0, 0.5, curve: Curves.easeIn),
      ),
    );

    // Start the animation sequence
    _startEmojiSequence();

    // Navigate to Home after 3 seconds
    Timer(const Duration(seconds: 3), () {
      _navigateToHome();
    });
  }

  void _startEmojiSequence() {
    // Show the first emoji immediately
    _animateEmoji();

    // Cycle through emojis every ~400ms to fit all 7 within 3 seconds
    _emojiTimer = Timer.periodic(const Duration(milliseconds: 400), (timer) {
      if (_currentIndex < _emojis.length - 1) {
        setState(() {
          _currentIndex++;
        });
        _controller.reset();
        _animateEmoji();
      } else {
        timer.cancel();
      }
    });
  }

  void _animateEmoji() {
    _controller.forward();
  }

  void _navigateToHome() {
    Get.off(
      () => const MainNavigation(),
      transition: Transition.fadeIn,
      duration: const Duration(milliseconds: 800),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    _emojiTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.lightBackground,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Centered content: Logo + Emoji Animation
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                // App Logo with fade-in animation
                FadeTransition(
                  opacity: _logoFadeAnimation,
                  child: Container(
                    width: 110.w,
                    height: 110.w,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(24.r),
                      boxShadow: [
                        BoxShadow(
                          color: AppColors.primary.withValues(alpha: 0.3),
                          blurRadius: 20,
                          spreadRadius: 2,
                          offset: const Offset(0, 6),
                        ),
                      ],
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(24.r),
                      child: Image.asset(
                        'assets/images/app_logo.png',
                        fit: BoxFit.cover,
                      ),
                    ),
                  ),
                ),

                SizedBox(height: 20.h),

                // Animated Emotion Emoji cycling through all 7 emotions
                ScaleTransition(
                  scale: _scaleAnimation,
                  child: FadeTransition(
                    opacity: _fadeAnimation,
                    child: Text(
                      _emojis[_currentIndex],
                      style: TextStyle(fontSize: 60.sp),
                    ),
                  ),
                ),
              ],
            ),
          ),

          // App Name and Tagline at the bottom
          Positioned(
            bottom: 150.h,
            left: 0,
            right: 0,
            child: Column(
              children: [
                Text(
                  AppStrings.appName,
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 28.sp,
                    fontWeight: FontWeight.bold,
                    color: AppColors.primary,
                    letterSpacing: 1.2,
                  ),
                ),
                SizedBox(height: 8.h),
                Text(
                  AppStrings.appTagline,
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 14.sp,
                    color: AppColors.textSecondary,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
