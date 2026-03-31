import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import '../../data/constants/colors.dart';
import '../../data/constants/strings.dart';
import '../../data/controllers/emotion_controller.dart';
import '../../data/controllers/navigation_controller.dart';
import '../../data/controllers/theme_controller.dart';
import '../../widgets/custom_text.dart';

import 'widgets/emotion_overview_card.dart';
import 'widgets/current_mood_card.dart';
import '../profile/trusted_contacts_screen.dart';
import '../notes/notes_screen.dart';
import '../videos/videos_screen.dart';

/// Enhanced Home screen - Main dashboard with current mood and insights
class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final emotionController = Get.find<EmotionController>();
    final navController = Get.find<NavigationController>();
    final themeController = Get.find<ThemeController>();
    return Obx(() {
      final _ = themeController.themeMode.value;

      return Scaffold(
        backgroundColor: Theme.of(context).scaffoldBackgroundColor,
        body: SafeArea(
          bottom: false,
          child: RefreshIndicator(
            onRefresh: () async {
              emotionController.refreshStats();
              await Future.delayed(const Duration(milliseconds: 500));
            },
            color: AppColors.primary,
            child: SingleChildScrollView(
              physics: const BouncingScrollPhysics(), // Smooth scrolling
              padding: EdgeInsets.symmetric(horizontal: 24.w),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SizedBox(height: 32.h),

                  // 1. Massive Minimal Header
                  CustomText(
                    AppStrings.homeWelcome,
                    fontSize: 42, // Very Large
                    fontWeight: FontWeight.w900,
                    color: AppColors.textPrimary,
                    letterSpacing: -1.5, // Tight tracking for impact
                    height: 1.1,
                  ),
                  SizedBox(height: 8.h),
                  CustomText(
                    AppStrings.homeGreeting, // "How are you feeling?"
                    fontSize: 18,
                    fontWeight: FontWeight.w400,
                    color: AppColors.textSecondary,
                    height: 1.4,
                  ),

                  SizedBox(height: 48.h), // Generous whitespace
                  // 2. Current Mood - Ultra Clean
                  CurrentMoodCard(
                    onCheckIn: () => navController.navigateToCamera(),
                  ),

                  SizedBox(height: 48.h),

                  // 3. Quick Actions - Simple Row
                  CustomText(
                    AppStrings.homeQuickActions,
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: AppColors.textPrimary,
                    letterSpacing: -0.5,
                  ),
                  SizedBox(height: 24.h),
                  // Row 1 – 3 equal-width buttons
                  Row(
                    children: [
                      Expanded(
                        child: _buildMinimalAction(
                          icon: Icons.chat_bubble_outline_rounded,
                          label: "Chat",
                          color: AppColors.primary,
                          onTap: () => navController.changeIndex(1),
                        ),
                      ),
                      Expanded(
                        child: _buildMinimalAction(
                          icon: Icons.camera_alt_outlined,
                          label: "Camera",
                          color: AppColors.secondary,
                          onTap: () => navController.navigateToCamera(),
                        ),
                      ),
                      Expanded(
                        child: _buildMinimalAction(
                          icon: Icons.connect_without_contact_sharp,
                          label: "Contacts",
                          color: AppColors.accentCoral,
                          onTap: () =>
                              Get.to(() => const TrustedContactsScreen()),
                        ),
                      ),
                    ],
                  ),
                  SizedBox(height: 24.h),
                  // Row 2 – 3 equal-width buttons
                  Row(
                    children: [
                      Expanded(
                        child: _buildMinimalAction(
                          icon: Icons.history_rounded,
                          label: "History",
                          color: AppColors.accent,
                          onTap: () => navController.changeIndex(3),
                        ),
                      ),
                      Expanded(
                        child: _buildMinimalAction(
                          icon: Icons.note_alt_outlined,
                          label: "Notes",
                          color: AppColors.accentMint,
                          onTap: () => Get.to(() => const NotesScreen()),
                        ),
                      ),
                      Expanded(
                        child: _buildMinimalAction(
                          icon: Icons.play_circle_filled,
                          label: "Videos",
                          color: AppColors.primaryBlue,
                          onTap: () => Get.to(() => const VideosScreen()),
                        ),
                      ),
                    ],
                  ),

                  SizedBox(height: 48.h),

                  // 4. Recent Emotions - Clean List
                  EmotionOverviewCard(),

                  SizedBox(height: 120.h), // Bottom padding
                ],
              ),
            ),
          ),
        ),
      );
    });
  }

  // Ultra-Minimal Action Button
  Widget _buildMinimalAction({
    required IconData icon,
    required String label,
    required Color color,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        children: [
          Container(
            width: 72.w, // Reduced from 80.w to prevent 'merged' look
            height: 72.w,
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.12),
              borderRadius: BorderRadius.circular(22.r),
              border: Border.all(color: color.withValues(alpha: 0.1), width: 1),
              boxShadow: [
                BoxShadow(
                  color: color.withValues(alpha: 0.05),
                  blurRadius: 12,
                  offset: const Offset(0, 4),
                ),
              ],
            ),
            child: Icon(icon, color: color, size: 28.sp),
          ),
          SizedBox(height: 10.h),
          CustomText(
            label,
            fontSize: 13,
            fontWeight: FontWeight.w600,
            color: AppColors.textPrimary,
          ),
        ],
      ),
    );
  }
}
