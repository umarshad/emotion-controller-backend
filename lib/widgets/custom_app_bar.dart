import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import '../data/constants/colors.dart';
import 'custom_text.dart';

/// Consistent app bar across screens
class CustomAppBar extends StatelessWidget implements PreferredSizeWidget {
  final String title;
  final List<Widget>? actions;
  final bool showBackButton;
  final VoidCallback? onBackPressed;
  final Widget? leading;

  const CustomAppBar({
    super.key,
    required this.title,
    this.actions,
    this.showBackButton = true,
    this.onBackPressed,
    this.leading,
  });

  @override
  Widget build(BuildContext context) {
    return AppBar(
      backgroundColor: Theme.of(context).colorScheme.surface,
      elevation: 0,
      scrolledUnderElevation: 0,
      surfaceTintColor: Colors.transparent,
      leading: showBackButton
          ? (leading ??
              IconButton(
                icon: Icon(
                  Icons.arrow_back_ios,
                  color: Theme.of(context).iconTheme.color ?? AppColors.textPrimary,
                  size: 20.sp,
                ),
                onPressed: onBackPressed ?? () => Navigator.of(context).pop(),
              ))
          : null,
      title: CustomText(
        title,
        fontSize: 20,
        fontWeight: FontWeight.bold,
        color: Theme.of(context).textTheme.titleLarge?.color ?? AppColors.textPrimary,
      ),
      actions: actions,
      centerTitle: false,
    );
  }

  @override
  Size get preferredSize => Size.fromHeight(56.h);
}