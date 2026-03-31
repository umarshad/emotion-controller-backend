import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import '../data/constants/colors.dart';
import '../data/controllers/navigation_controller.dart';
import '../data/controllers/theme_controller.dart';
import '../widgets/breathing_camera_button.dart';
import 'home/home_screen.dart';
import 'chat/chat_screen.dart';
import 'camera/camera_screen.dart';
import 'history/emotion_history_screen.dart';
import 'profile/profile_screen.dart';

/// Main navigation with persistent bottom nav bar and center-docked camera button
class MainNavigation extends StatelessWidget {
  const MainNavigation({super.key});

  @override
  Widget build(BuildContext context) {
    final navController = Get.find<NavigationController>();
    final themeController = Get.find<ThemeController>();

    return Obx(() {
      // Access observable to make this reactive to theme changes
      final _ = themeController.themeMode.value;
      return Scaffold(
        body: SafeArea(
          bottom: false, // Bottom safe area handled by nav bar
          child: IndexedStack(
            index: navController.currentIndex.value,
            children: const [
              HomeScreen(),
              ChatScreen(),
              CameraScreen(),
              EmotionHistoryScreen(),
              ProfileScreen(),
            ],
          ),
        ),
        bottomNavigationBar: _buildBottomNavBar(context, navController),
      );
    });
  }

  Widget _buildBottomNavBar(
    BuildContext context,
    NavigationController navController,
  ) {
    return SafeArea(
      top: false,
      child: Container(
        height: 70.h,
        padding: EdgeInsets.only(bottom: 0, top: 4.h, left: 8.w, right: 8.w),
        decoration: BoxDecoration(
          color: Theme.of(context).colorScheme.surface,
          borderRadius: BorderRadius.only(
            topLeft: Radius.circular(20.r),
            topRight: Radius.circular(20.r),
          ),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            _buildNavItem(
              icon: Icons.home_outlined,
              activeIcon: Icons.home,
              label: 'Home',
              index: 0,
              navController: navController,
            ),
            _buildNavItem(
              icon: Icons.chat_bubble_outline,
              activeIcon: Icons.chat_bubble,
              label: 'Chat',
              index: 1,
              navController: navController,
            ),
            _buildCameraButton(navController),
            _buildNavItem(
              icon: Icons.history_outlined,
              activeIcon: Icons.history,
              label: 'History',
              index: 3,
              navController: navController,
            ),
            _buildNavItem(
              icon: Icons.format_quote_outlined,
              activeIcon: Icons.format_quote,
              label: 'Quotes',
              index: 4,
              navController: navController,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildNavItem({
    required IconData icon,
    required IconData activeIcon,
    required String label,
    required int index,
    required NavigationController navController,
  }) {
    final isSelected = navController.currentIndex.value == index;

    return GestureDetector(
      onTap: () => navController.changeIndex(index),
      child: Container(
        padding: EdgeInsets.symmetric(horizontal: 12.w, vertical: 8.h),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            AnimatedSwitcher(
              duration: const Duration(milliseconds: 200),
              child: Icon(
                isSelected ? activeIcon : icon,
                key: ValueKey(isSelected),
                color: isSelected ? AppColors.primary : AppColors.textLight,
                size: 24.sp,
              ),
            ),
            SizedBox(height: 4.h),
            Text(
              label,
              style: TextStyle(
                fontSize: 10.sp,
                color: isSelected ? AppColors.primary : AppColors.textLight,
                fontWeight: isSelected ? FontWeight.w600 : FontWeight.normal,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCameraButton(NavigationController navController) {
    final isSelected = navController.currentIndex.value == 2;

    return BreathingCameraButton(
      onTap: () => navController.navigateToCamera(),
      isSelected: isSelected,
    );
  }
}
