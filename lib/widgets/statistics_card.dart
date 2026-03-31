import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import '../data/constants/colors.dart';
import 'custom_text.dart';

/// Reusable statistics card widget
class StatisticsCard extends StatelessWidget {
  final String title;
  final String value;
  final String? subtitle;
  final IconData? icon;
  final Color? iconColor;
  final VoidCallback? onTap;

  const StatisticsCard({
    super.key,
    required this.title,
    required this.value,
    this.subtitle,
    this.icon,
    this.iconColor,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: EdgeInsets.all(20.w),
        decoration: BoxDecoration(
          color: Theme.of(context).cardColor,
          borderRadius: BorderRadius.circular(16.r),
          border: Border.all(color: Theme.of(context).dividerColor),
          boxShadow: [
            BoxShadow(
              color: Theme.of(context).shadowColor,
              blurRadius: 8,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Expanded(
                  child: CustomText(
                    title,
                    fontSize: 14,
                    color:
                        Theme.of(context).textTheme.bodySmall?.color ??
                        AppColors.textSecondary,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                if (icon != null)
                  Container(
                    padding: EdgeInsets.all(8.w),
                    decoration: BoxDecoration(
                      color: (iconColor ?? AppColors.primary).withValues(
                        alpha: 0.1,
                      ),
                      borderRadius: BorderRadius.circular(8.r),
                    ),
                    child: Icon(
                      icon,
                      size: 20.sp,
                      color: iconColor ?? AppColors.primary,
                    ),
                  ),
              ],
            ),
            SizedBox(height: 12.h),
            CustomText(
              value,
              fontSize: 32,
              fontWeight: FontWeight.bold,
              color:
                  Theme.of(context).textTheme.bodyLarge?.color ??
                  AppColors.textPrimary,
            ),
            if (subtitle != null) ...[
              SizedBox(height: 4.h),
              CustomText(
                subtitle!,
                fontSize: 12,
                color:
                    Theme.of(context).textTheme.bodySmall?.color ??
                    AppColors.textLight,
              ),
            ],
          ],
        ),
      ),
    );
  }
}
