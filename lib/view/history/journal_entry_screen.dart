import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import '../../data/constants/colors.dart';
import '../../data/controllers/emotion_controller.dart';
import '../../data/models/journal_model.dart';
import '../../widgets/custom_text.dart';
import '../../widgets/custom_app_bar.dart';

class JournalEntryScreen extends StatefulWidget {
  final String emotionId;
  final String prompt;
  final String? existingContent;

  const JournalEntryScreen({
    super.key,
    required this.emotionId,
    required this.prompt,
    this.existingContent,
  });

  @override
  State<JournalEntryScreen> createState() => _JournalEntryScreenState();
}

class _JournalEntryScreenState extends State<JournalEntryScreen> {
  late TextEditingController _controller;
  final _emotionController = Get.find<EmotionController>();

  @override
  void initState() {
    super.initState();
    _controller = TextEditingController(text: widget.existingContent);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _saveEntry() {
    if (_controller.text.trim().isEmpty) {
      Get.snackbar(
        'Empty Entry',
        'Please write something before saving.',
        snackPosition: SnackPosition.BOTTOM,
        backgroundColor: Colors.redAccent,
        colorText: Colors.white,
      );
      return;
    }

    final entry = JournalModel(
      id: DateTime.now().toIso8601String(),
      emotionId: widget.emotionId,
      prompt: widget.prompt,
      content: _controller.text.trim(),
      timestamp: DateTime.now(),
    );

    _emotionController.addJournalEntry(entry);
    Get.back();
    Get.snackbar(
      'Saved',
      'Your reflection has been saved.',
      snackPosition: SnackPosition.BOTTOM,
      backgroundColor: AppColors.primary,
      colorText: Colors.white,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      appBar: CustomAppBar(title: 'Journal Entry', showBackButton: true),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(20.w),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              padding: EdgeInsets.all(16.w),
              decoration: BoxDecoration(
                color: AppColors.primary.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(12.r),
                border: Border.all(
                  color: AppColors.primary.withValues(alpha: 0.2),
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        Icons.lightbulb_outline,
                        color: AppColors.primary,
                        size: 20.sp,
                      ),
                      SizedBox(width: 8.w),
                      CustomText(
                        'Reflection Prompt',
                        fontSize: 14,
                        fontWeight: FontWeight.bold,
                        color: AppColors.primary,
                      ),
                    ],
                  ),
                  SizedBox(height: 8.h),
                  CustomText(
                    widget.prompt,
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                    height: 1.4,
                  ),
                ],
              ),
            ),
            SizedBox(height: 32.h),
            CustomText(
              'Your Reflection',
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
            SizedBox(height: 12.h),
            TextField(
              controller: _controller,
              maxLines: 10,
              decoration: InputDecoration(
                hintText: 'Start writing here...',
                filled: true,
                fillColor: Theme.of(context).cardColor,
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12.r),
                  borderSide: BorderSide(color: Theme.of(context).dividerColor),
                ),
                enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12.r),
                  borderSide: BorderSide(color: Theme.of(context).dividerColor),
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12.r),
                  borderSide: BorderSide(color: AppColors.primary),
                ),
              ),
              style: TextStyle(
                fontSize: 16.sp,
                height: 1.5,
                color: Theme.of(context).textTheme.bodyLarge?.color,
              ),
            ),
            SizedBox(height: 32.h),
            SizedBox(
              width: double.infinity,
              height: 56.h,
              child: ElevatedButton(
                onPressed: _saveEntry,
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.primary,
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12.r),
                  ),
                  elevation: 0,
                ),
                child: CustomText(
                  'Save Reflection',
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
