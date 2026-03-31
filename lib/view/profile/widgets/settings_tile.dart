import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import '../../../data/constants/colors.dart';
import '../../../widgets/custom_text.dart';

/// Reusable settings option widget
class SettingsTile extends StatelessWidget {
  final String title;
  final String? subtitle;
  final IconData? icon;
  final VoidCallback? onTap;
  final Widget? trailing;
  final bool showDivider;

  const SettingsTile({
    super.key,
    required this.title,
    this.subtitle,
    this.icon,
    this.onTap,
    this.trailing,
    this.showDivider = true,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        InkWell(
          onTap: onTap,
          child: Container(
            padding: EdgeInsets.symmetric(horizontal: 16.w, vertical: 16.h),
            child: Row(
              children: [
                if (icon != null) ...[
                  Container(
                    width: 40.w,
                    height: 40.h,
                    decoration: BoxDecoration(
                      color: AppColors.primary.withValues(alpha: 0.1),
                      borderRadius: BorderRadius.circular(10.r),
                    ),
                    child: Icon(icon, color: AppColors.primary, size: 20.sp),
                  ),
                  SizedBox(width: 16.w),
                ],
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      CustomText(
                        title,
                        fontSize: 16,
                        fontWeight: FontWeight.w500,
                      ),
                      if (subtitle != null) ...[
                        SizedBox(height: 4.h),
                        CustomText(
                          subtitle!,
                          fontSize: 12,
                          color:
                              Theme.of(context).textTheme.bodySmall?.color ??
                              AppColors.textSecondary,
                        ),
                      ],
                    ],
                  ),
                ),
                if (trailing != null) trailing!,
                if (onTap != null && trailing == null)
                  Icon(
                    Icons.chevron_right,
                    color:
                        Theme.of(context).textTheme.bodySmall?.color ??
                        AppColors.textLight,
                    size: 20.sp,
                  ),
              ],
            ),
          ),
        ),
        if (showDivider)
          Divider(
            height: 1,
            color: Theme.of(context).dividerColor,
            indent: icon != null ? 72.w : 16.w,
          ),
      ],
    );
  }
}
