import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import '../../../data/constants/colors.dart';
import '../../../widgets/custom_text.dart';
import 'quote_detail_bottom_sheet.dart';

/// Bottom sheet displaying all motivational quotes
class MotivationalBottomSheet extends StatelessWidget {
  final List<String> quotes;
  final Function(String)? onQuoteSelected;

  const MotivationalBottomSheet({
    super.key,
    required this.quotes,
    this.onQuoteSelected,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.only(
        bottom: MediaQuery.of(context).viewInsets.bottom,
      ),
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
            margin: EdgeInsets.only(top: 12.h, bottom: 24.h),
            decoration: BoxDecoration(
              color: Theme.of(context).dividerColor,
              borderRadius: BorderRadius.circular(2.r),
            ),
          ),
          // Title
          Padding(
            padding: EdgeInsets.symmetric(horizontal: 24.w),
            child: Row(
              children: [
                Icon(
                  Icons.lightbulb_outline,
                  color: AppColors.primary,
                  size: 24.sp,
                ),
                SizedBox(width: 12.w),
                CustomText(
                  'Daily Inspiration',
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ],
            ),
          ),
          SizedBox(height: 24.h),
          // Quotes list
          Flexible(
            child: ListView.builder(
              shrinkWrap: true,
              padding: EdgeInsets.symmetric(horizontal: 24.w),
              itemCount: quotes.length,
              itemBuilder: (context, index) {
                final quote = quotes[index];
                return GestureDetector(
                  onTap: () {
                    // Close current sheet using GetX
                    Get.back();

                    // Show quote detail sheet using GetX (more stable than context-based)
                    Get.bottomSheet(
                      QuoteDetailBottomSheet(
                        selectedQuote: quote,
                        allQuotes: quotes,
                      ),
                      backgroundColor: Colors.transparent,
                      isScrollControlled: true,
                    );
                  },
                  child: Container(
                    margin: EdgeInsets.only(bottom: 16.h),
                    padding: EdgeInsets.all(16.w),
                    decoration: BoxDecoration(
                      color: Theme.of(context).colorScheme.surface,
                      borderRadius: BorderRadius.circular(12.r),
                      border: Border.all(color: Theme.of(context).dividerColor),
                    ),
                    child: Row(
                      children: [
                        Expanded(
                          child: CustomText(
                            quote,
                            fontSize: 16,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                        Icon(
                          Icons.arrow_forward_ios,
                          size: 16.sp,
                          color:
                              Theme.of(context).textTheme.bodySmall?.color ??
                              AppColors.textLight,
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
          SizedBox(height: 24.h),
        ],
      ),
    );
  }
}
