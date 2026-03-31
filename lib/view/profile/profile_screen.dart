import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import '../../data/constants/colors.dart';
import '../../data/constants/quotes.dart';
import '../../data/controllers/theme_controller.dart';
import '../../widgets/custom_text.dart';
import '../../widgets/custom_app_bar.dart';
import 'trusted_contacts_screen.dart';

/// Screen displaying quotes for trained emotions
class ProfileScreen extends StatelessWidget {
  const ProfileScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final themeController = Get.find<ThemeController>();
    final List<String> emotions = Quotes.data.keys.toList();

    return Obx(() {
      // Access observable to make this reactive
      final _ = themeController.themeMode.value;
      return Scaffold(
        backgroundColor: Theme.of(context).scaffoldBackgroundColor,
        appBar: CustomAppBar(title: "Emotion Quotes", showBackButton: false),
        body: ListView.separated(
          padding: EdgeInsets.fromLTRB(16.w, 16.h, 16.w, 90.h),
          itemCount: emotions.length,
          separatorBuilder: (context, index) => SizedBox(height: 16.h),
          itemBuilder: (context, index) {
            final emotion = emotions[index];
            final quotes = Quotes.data[emotion]!;
            final color = _getEmotionColor(emotion);

            return Column(
              children: [
                Container(
                  decoration: BoxDecoration(
                    color: Theme.of(context).cardColor,
                    borderRadius: BorderRadius.circular(16.r),
                    border: Border.all(color: color.withValues(alpha: 0.3)),
                    boxShadow: [
                      BoxShadow(
                        color: color.withValues(alpha: 0.05),
                        blurRadius: 10,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: ExpansionTile(
                    leading: Container(
                      padding: EdgeInsets.all(8.w),
                      decoration: BoxDecoration(
                        color: color.withValues(alpha: 0.1),
                        shape: BoxShape.circle,
                      ),
                      child: Icon(
                        Icons.format_quote,
                        color: color,
                        size: 20.sp,
                      ),
                    ),
                    title: CustomText(
                      emotion.toUpperCase(),
                      fontWeight: FontWeight.bold,
                      fontSize: 16,
                      color:
                          Theme.of(context).textTheme.bodyLarge?.color ??
                          AppColors.textPrimary,
                    ),
                    children: quotes
                        .map(
                          (quote) => Padding(
                            padding: EdgeInsets.symmetric(
                              horizontal: 16.w,
                              vertical: 8.h,
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                CustomText(
                                  "\"$quote\"",
                                  fontSize: 14,
                                  fontStyle: FontStyle.italic,
                                  color:
                                      Theme.of(
                                        context,
                                      ).textTheme.bodyMedium?.color ??
                                      AppColors.textSecondary,
                                  height: 1.4,
                                ),
                                SizedBox(height: 8.h),
                                Divider(
                                  color: Theme.of(
                                    context,
                                  ).dividerColor.withValues(alpha: 0.5),
                                ),
                              ],
                            ),
                          ),
                        )
                        .toList(),
                  ),
                ),
                if (index == emotions.length - 1) ...[
                  SizedBox(height: 32.h),
                  _buildLovedOnesButton(context),
                ],
              ],
            );
          },
        ),
      );
    });
  }

  Widget _buildLovedOnesButton(BuildContext context) {
    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [AppColors.primary, AppColors.primary.withValues(alpha: 0.8)],
        ),
        borderRadius: BorderRadius.circular(16.r),
        boxShadow: [
          BoxShadow(
            color: AppColors.primary.withValues(alpha: 0.3),
            blurRadius: 12,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: () => Get.to(() => const TrustedContactsScreen()),
          borderRadius: BorderRadius.circular(16.r),
          child: Padding(
            padding: EdgeInsets.symmetric(vertical: 16.h, horizontal: 20.w),
            child: Row(
              children: [
                Icon(Icons.favorite, color: Colors.white, size: 24.sp),
                SizedBox(width: 16.w),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      CustomText(
                        "Manage Loved Ones",
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 16,
                      ),
                      CustomText(
                        "Add trusted contacts for support",
                        color: Colors.white.withValues(alpha: 0.9),
                        fontSize: 12,
                      ),
                    ],
                  ),
                ),
                const Icon(
                  Icons.arrow_forward_ios,
                  color: Colors.white,
                  size: 16,
                ),
              ],
            ),
          ),
        ),
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
      case 'fear':
        return const Color(0xFF9C27B0);
      case 'surprise':
        return const Color(0xFFFF9800);
      default:
        return const Color(0xFF4CAF50);
    }
  }
}
