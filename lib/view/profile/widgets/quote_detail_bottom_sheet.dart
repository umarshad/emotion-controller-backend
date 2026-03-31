import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import 'dart:math';
import '../../../data/constants/colors.dart';
import '../../../widgets/custom_text.dart';

/// Bottom sheet displaying a single quote with related/suggestive quotes
class QuoteDetailBottomSheet extends StatelessWidget {
  final String selectedQuote;
  final List<String> allQuotes;

  const QuoteDetailBottomSheet({
    super.key,
    required this.selectedQuote,
    required this.allQuotes,
  });

  /// Get 4-5 random related quotes (excluding the selected one)
  List<String> _getRelatedQuotes() {
    final random = Random();
    final otherQuotes = allQuotes.where((q) => q != selectedQuote).toList();

    // Shuffle and take 4-5 quotes
    otherQuotes.shuffle(random);
    final count = min(5, otherQuotes.length);
    return otherQuotes.take(count).toList();
  }

  @override
  Widget build(BuildContext context) {
    final relatedQuotes = _getRelatedQuotes();

    return Container(
      constraints: BoxConstraints(maxHeight: Get.height * 0.85),
      decoration: BoxDecoration(
        color: Theme.of(context).cardColor,
        borderRadius: BorderRadius.only(
          topLeft: Radius.circular(24.r),
          topRight: Radius.circular(24.r),
        ),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Handle
          Container(
            width: 40.w,
            height: 4.h,
            margin: EdgeInsets.only(top: 12.h, bottom: 20.h),
            decoration: BoxDecoration(
              color: Theme.of(context).dividerColor,
              borderRadius: BorderRadius.circular(2.r),
            ),
          ),

          // Scrollable content
          Flexible(
            child: SingleChildScrollView(
              padding: EdgeInsets.symmetric(horizontal: 24.w),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // Main Quote Card
                  Container(
                    width: double.infinity,
                    padding: EdgeInsets.all(24.w),
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [
                          AppColors.primary.withValues(alpha: 0.12),
                          AppColors.primary.withValues(alpha: 0.06),
                        ],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      borderRadius: BorderRadius.circular(20.r),
                      border: Border.all(
                        color: AppColors.primary.withValues(alpha: 0.3),
                        width: 2.w,
                      ),
                    ),
                    child: Column(
                      children: [
                        // Quote icon & Actions
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            SizedBox(width: 32.w), // Spacer for centering
                            Container(
                              width: 48.w,
                              height: 48.h,
                              decoration: BoxDecoration(
                                color: AppColors.primary.withValues(alpha: 0.2),
                                shape: BoxShape.circle,
                              ),
                              child: Icon(
                                Icons.format_quote,
                                color: AppColors.primary,
                                size: 28.sp,
                              ),
                            ),
                            IconButton(
                              onPressed: () {
                                Clipboard.setData(
                                  ClipboardData(text: selectedQuote),
                                ).then((_) {
                                  Get.snackbar(
                                    'Success',
                                    'Quote copied to clipboard',
                                    snackPosition: SnackPosition.BOTTOM,
                                    backgroundColor: AppColors.primary
                                        .withValues(alpha: 0.9),
                                    colorText: Colors.white,
                                    margin: EdgeInsets.all(16.w),
                                    duration: const Duration(seconds: 2),
                                  );
                                });
                              },
                              icon: Icon(
                                Icons.copy_rounded,
                                color: AppColors.primary,
                                size: 20.sp,
                              ),
                              tooltip: 'Copy Quote',
                            ),
                          ],
                        ),
                        SizedBox(height: 16.h),
                        // Quote text
                        CustomText(
                          selectedQuote,
                          fontSize: 18,
                          fontWeight: FontWeight.w600,
                          textAlign: TextAlign.center,
                          color:
                              Theme.of(context).textTheme.bodyLarge?.color ??
                              AppColors.textPrimary,
                        ),
                      ],
                    ),
                  ),

                  SizedBox(height: 24.h),

                  // Related Quotes Section
                  if (relatedQuotes.isNotEmpty) ...[
                    Row(
                      children: [
                        Icon(
                          Icons.lightbulb_outline,
                          color: AppColors.primary,
                          size: 20.sp,
                        ),
                        SizedBox(width: 8.w),
                        CustomText(
                          'Suggestion Quotes',
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color:
                              Theme.of(context).textTheme.bodyMedium?.color ??
                              AppColors.textSecondary,
                        ),
                      ],
                    ),
                    SizedBox(height: 16.h),

                    // Related quotes list using Column for nesting safety
                    ...relatedQuotes.asMap().entries.map((entry) {
                      final index = entry.key;
                      final quote = entry.value;
                      return TweenAnimationBuilder<double>(
                        tween: Tween(begin: 0.0, end: 1.0),
                        duration: Duration(milliseconds: 300 + (index * 100)),
                        curve: Curves.easeOut,
                        builder: (context, value, child) {
                          return Opacity(
                            opacity: value,
                            child: Transform.translate(
                              offset: Offset(0, 20 * (1 - value)),
                              child: child,
                            ),
                          );
                        },
                        child: GestureDetector(
                          onTap: () {
                            // Use GetX for stable transitions
                            Get.back();
                            Get.bottomSheet(
                              QuoteDetailBottomSheet(
                                selectedQuote: quote,
                                allQuotes: allQuotes,
                              ),
                              backgroundColor: Colors.transparent,
                              isScrollControlled: true,
                            );
                          },
                          child: Container(
                            margin: EdgeInsets.only(bottom: 12.h),
                            padding: EdgeInsets.all(16.w),
                            decoration: BoxDecoration(
                              color: Theme.of(context).colorScheme.surface,
                              borderRadius: BorderRadius.circular(12.r),
                              border: Border.all(
                                color: Theme.of(context).dividerColor,
                              ),
                            ),
                            child: Row(
                              children: [
                                Container(
                                  width: 6.w,
                                  height: 40.h,
                                  decoration: BoxDecoration(
                                    color: AppColors.primary.withValues(
                                      alpha: 0.6,
                                    ),
                                    borderRadius: BorderRadius.circular(3.r),
                                  ),
                                ),
                                SizedBox(width: 12.w),
                                Expanded(
                                  child: CustomText(
                                    quote,
                                    fontSize: 14,
                                    fontWeight: FontWeight.w500,
                                    color:
                                        Theme.of(
                                          context,
                                        ).textTheme.bodyMedium?.color ??
                                        AppColors.textSecondary,
                                  ),
                                ),
                                Icon(
                                  Icons.arrow_forward_ios,
                                  size: 14.sp,
                                  color:
                                      Theme.of(
                                        context,
                                      ).textTheme.bodySmall?.color ??
                                      AppColors.textLight,
                                ),
                              ],
                            ),
                          ),
                        ),
                      );
                    }),
                  ],
                ],
              ),
            ),
          ),

          SizedBox(height: 16.h),

          // Close button
          Padding(
            padding: EdgeInsets.only(left: 24.w, right: 24.w, bottom: 24.h),
            child: SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () => Get.back(),
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.primary,
                  padding: EdgeInsets.symmetric(vertical: 16.h),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12.r),
                  ),
                  elevation: 0,
                ),
                child: CustomText(
                  'Close',
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: Colors.white,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
