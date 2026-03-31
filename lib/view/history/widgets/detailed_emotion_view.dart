import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import '../../../data/constants/colors.dart';
import '../../../data/constants/assets.dart';
import '../../../data/controllers/theme_controller.dart';
import '../../../data/controllers/navigation_controller.dart';
import '../../../data/models/emotion_model.dart';
import '../../../data/utils/helpers.dart';
import '../../../widgets/custom_text.dart';
import '../../../widgets/emotion_video_player.dart';
import '../../../data/controllers/emotion_controller.dart';
import '../../camera/quotes_detail_screen.dart';
import '../../profile/trusted_contacts_screen.dart';
import '../journal_entry_screen.dart';
import '../../../widgets/emotion_rating_dialog.dart';

/// Detailed emotion view screen
class DetailedEmotionView extends StatelessWidget {
  final EmotionModel emotion;

  final VoidCallback? onSave;

  const DetailedEmotionView({super.key, required this.emotion, this.onSave});

  @override
  Widget build(BuildContext context) {
    final emotionColor = getEmotionColor(emotion.type);
    final emotionName = AppAssets.getEmotionName(emotion.type);
    final emoji = AppAssets.getEmotionEmoji(emotion.type);
    final themeController = Get.find<ThemeController>();

    return Obx(() {
      // Access observable to make this reactive
      final _ = themeController.themeMode.value;
      return Scaffold(
        backgroundColor: Theme.of(context).scaffoldBackgroundColor,
        appBar: AppBar(
          backgroundColor: Theme.of(context).colorScheme.surface,
          elevation: 0,
          leading: IconButton(
            icon: Icon(
              Icons.arrow_back,
              color: Theme.of(context).iconTheme.color ?? AppColors.textPrimary,
            ),
            onPressed: () {
              // If onSave is provided (preview mode), just go back to preview
              // Otherwise go to home
              if (onSave != null) {
                Get.back();
              } else {
                Get.until((route) => route.isFirst);
                try {
                  Get.find<NavigationController>().changeIndex(0);
                } catch (e) {
                  // Fallback if NavigationController not found
                  Get.back();
                }
              }
            },
          ),
          title: CustomText(
            'Emotion Details',
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),
        body: SingleChildScrollView(
          padding: EdgeInsets.all(20.w),
          child: Column(
            children: [
              // Large emotion display
              Container(
                padding: EdgeInsets.all(32.w),
                decoration: BoxDecoration(
                  color: emotionColor.withValues(alpha: 0.1),
                  shape: BoxShape.circle,
                ),
                child: Text(emoji, style: TextStyle(fontSize: 80.sp)),
              ),
              SizedBox(height: 24.h),
              CustomText(
                emotionName,
                fontSize: 32,
                fontWeight: FontWeight.bold,
                color: emotionColor,
              ),
              SizedBox(height: 8.h),
              CustomText(
                '${emotion.intensity}% • ${Helpers.getIntensityLevel(emotion.intensity)}',
                fontSize: 16,
                color:
                    Theme.of(context).textTheme.bodyMedium?.color ??
                    AppColors.textSecondary,
              ),
              SizedBox(height: 32.h),

              // Details card
              Container(
                padding: EdgeInsets.all(20.w),
                decoration: BoxDecoration(
                  color: Theme.of(context).cardColor,
                  borderRadius: BorderRadius.circular(16.r),
                  border: Border.all(color: Theme.of(context).dividerColor),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _buildDetailRow(
                      'Detection Method',
                      emotion.detectionMethod == 'chat' ? 'AI Chat' : 'Camera',
                    ),
                    SizedBox(height: 16.h),
                    Divider(color: Theme.of(context).dividerColor),
                    SizedBox(height: 16.h),
                    _buildDetailRow(
                      'Date & Time',
                      Helpers.formatDateTime(emotion.timestamp),
                    ),
                    SizedBox(height: 16.h),
                    Divider(color: Theme.of(context).dividerColor),
                    SizedBox(height: 16.h),
                    _buildDetailRow(
                      'Intensity Level',
                      Helpers.getIntensityLevel(emotion.intensity),
                    ),
                    SizedBox(height: 16.h),
                    Divider(color: Theme.of(context).dividerColor),
                    SizedBox(height: 16.h),
                    Obx(() {
                      final emotionController = Get.find<EmotionController>();
                      // Find the latest version of this emotion in the controller
                      final currentEmotionState = emotionController.emotions
                          .firstWhereOrNull((e) => e.id == emotion.id);

                      // Use current state if found, otherwise fallback to the passed emotion
                      final displayRating =
                          currentEmotionState?.userRating ?? emotion.userRating;

                      return Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          CustomText('Accuracy Rating',
                              fontSize: 14, color: AppColors.textSecondary),
                          GestureDetector(
                            onTap: () {
                              Get.dialog(
                                EmotionRatingDialog(
                                  emotionId: emotion.id,
                                  initialRating: displayRating ?? 0,
                                  onRatingSelected: (rating) {
                                    emotionController.updateEmotionRating(
                                        emotion.id, rating);
                                  },
                                ),
                              );
                            },
                            child: Row(
                              children: [
                                if (displayRating != null && displayRating > 0)
                                  ...List.generate(
                                    5,
                                    (index) => Icon(
                                      index < displayRating
                                          ? Icons.star_rounded
                                          : Icons.star_outline_rounded,
                                      color: index < displayRating
                                          ? Colors.amber
                                          : AppColors.textLight
                                              .withValues(alpha: 0.5),
                                      size: 20.sp,
                                    ),
                                  )
                                else
                                  CustomText(
                                    'Rate Now',
                                    fontSize: 14,
                                    fontWeight: FontWeight.w600,
                                    color: AppColors.primary,
                                  ),
                                SizedBox(width: 4.w),
                                Icon(
                                  Icons.edit,
                                  size: 14.sp,
                                  color: AppColors.primary,
                                ),
                              ],
                            ),
                          ),
                        ],
                      );
                    }),
                  ],
                ),
              ),
              SizedBox(height: 24.h),

              // Intensity bar
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  CustomText(
                    'Intensity',
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                  ),
                  SizedBox(height: 8.h),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(8.r),
                    child: LinearProgressIndicator(
                      value: emotion.intensity / 100,
                      backgroundColor: emotionColor.withValues(alpha: 0.1),
                      valueColor: AlwaysStoppedAnimation<Color>(emotionColor),
                      minHeight: 12.h,
                    ),
                  ),
                ],
              ),
              SizedBox(height: 32.h),

              // Suggestions
              Align(
                alignment: Alignment.centerLeft,
                child: CustomText(
                  'Personalized Suggestions',
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
              SizedBox(height: 16.h),
              ...emotion.suggestions.map(
                (suggestion) => Padding(
                  padding: EdgeInsets.only(bottom: 12.h),
                  child: InkWell(
                    onTap: () {
                      final type = emotion.type.toLowerCase();
                      final text = suggestion.toLowerCase();
                      if (type == 'sad' &&
                          (text.contains('friend') ||
                              text.contains('loved one'))) {
                        Get.to(() => const TrustedContactsScreen());
                      } else {
                        Get.to(() => QuotesDetailScreen(emotion: emotion));
                      }
                    },
                    borderRadius: BorderRadius.circular(12.r),
                    child: Container(
                      padding: EdgeInsets.all(16.w),
                      decoration: BoxDecoration(
                        color: Theme.of(context).cardColor,
                        borderRadius: BorderRadius.circular(12.r),
                        border: Border.all(
                          color: Theme.of(context).dividerColor,
                        ),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withValues(alpha: 0.01),
                            blurRadius: 2,
                            offset: const Offset(0, 1),
                          ),
                        ],
                      ),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Icon(
                            Icons.lightbulb_outline,
                            color: emotionColor,
                            size: 24.sp,
                          ),
                          SizedBox(width: 12.w),
                          Expanded(
                            child: CustomText(
                              suggestion,
                              fontSize: 14,
                              height: 1.4,
                            ),
                          ),
                          SizedBox(width: 8.w),
                          Icon(
                            Icons.arrow_forward_ios_rounded,
                            size: 14.sp,
                            color: AppColors.textLight.withValues(alpha: 0.4),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
              SizedBox(height: 32.h),

              // Video player
              EmotionVideoPlayer(emotionType: emotion.type),
              SizedBox(height: 32.h),

              // Journaling Section
              Align(
                alignment: Alignment.centerLeft,
                child: CustomText(
                  'Reflection & Journaling',
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
              SizedBox(height: 16.h),
              Obx(() {
                final emotionController = Get.find<EmotionController>();
                final journalEntry = emotionController.getJournalForEmotion(
                  emotion.id,
                );

                if (journalEntry != null) {
                  return Container(
                    padding: EdgeInsets.all(20.w),
                    decoration: BoxDecoration(
                      color: emotionColor.withValues(alpha: 0.05),
                      borderRadius: BorderRadius.circular(16.r),
                      border: Border.all(
                        color: emotionColor.withValues(alpha: 0.2),
                      ),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Icon(
                              Icons.edit_note,
                              color: emotionColor,
                              size: 24.sp,
                            ),
                            SizedBox(width: 8.w),
                            CustomText(
                              'Your Reflection',
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                              color: emotionColor,
                            ),
                            const Spacer(),
                            CustomText(
                              Helpers.formatDate(journalEntry.timestamp),
                              fontSize: 12,
                              color: AppColors.textLight,
                            ),
                          ],
                        ),
                        SizedBox(height: 12.h),
                        CustomText(
                          journalEntry.content,
                          fontSize: 14,
                          height: 1.5,
                          fontStyle: FontStyle.italic,
                        ),
                        SizedBox(height: 16.h),
                        TextButton.icon(
                          onPressed: () {
                            // Navigate to JournalEntryScreen for editing
                            Get.to(
                              () => JournalEntryScreen(
                                emotionId: emotion.id,
                                prompt: journalEntry.prompt,
                                existingContent: journalEntry.content,
                              ),
                            );
                          },
                          icon: Icon(
                            Icons.edit,
                            size: 16.sp,
                            color: emotionColor,
                          ),
                          label: CustomText(
                            'Edit Entry',
                            fontSize: 14,
                            color: emotionColor,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                  );
                }

                return Container(
                  padding: EdgeInsets.all(20.w),
                  decoration: BoxDecoration(
                    color: Theme.of(context).cardColor,
                    borderRadius: BorderRadius.circular(16.r),
                    border: Border.all(color: Theme.of(context).dividerColor),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      CustomText(
                        emotion.journallingPrompt ??
                            "Take a moment to reflect on this feeling.",
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                        height: 1.4,
                      ),
                      SizedBox(height: 20.h),
                      SizedBox(
                        width: double.infinity,
                        child: ElevatedButton(
                          onPressed: () {
                            Get.to(
                              () => JournalEntryScreen(
                                emotionId: emotion.id,
                                prompt:
                                    emotion.journallingPrompt ??
                                    "How are you feeling about this moment?",
                              ),
                            );
                          },
                          style: ElevatedButton.styleFrom(
                            backgroundColor: emotionColor,
                            foregroundColor: Colors.white,
                            padding: EdgeInsets.symmetric(vertical: 12.h),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12.r),
                            ),
                            elevation: 0,
                          ),
                          child: const Text('Start Journaling'),
                        ),
                      ),
                    ],
                  ),
                );
              }),
              SizedBox(height: 80.h), // Extra space for FAB
            ],
          ),
        ),
        floatingActionButton: onSave != null
            // Save Button (New Detection)
            ? FloatingActionButton.extended(
                onPressed: onSave,
                label: const Text("Save Emotion"),
                icon: const Icon(Icons.check),
                backgroundColor: AppColors.primary,
              )
            // Retake Button (History View)
            : FloatingActionButton.extended(
                onPressed: () {
                  // RETAKE (Go to Camera)
                  Get.until((route) => route.isFirst);
                  try {
                    Get.find<NavigationController>().navigateToCamera();
                  } catch (e) {
                    // Fallback
                  }
                },
                label: const Text("Retake"),
                icon: const Icon(Icons.camera_alt),
                backgroundColor: AppColors.primary,
              ),
      );
    });
  }

  Widget _buildDetailRow(String label, String value) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        CustomText(label, fontSize: 14, color: AppColors.textSecondary),
        CustomText(value, fontSize: 14, fontWeight: FontWeight.w600),
      ],
    );
  }
}

/// Get emotion color helper
/// Uses the 7 backend emotions
Color getEmotionColor(String emotionType) {
  switch (emotionType.toLowerCase()) {
    case 'angry':
      return AppColors.emotionAngry;
    case 'disgust':
      return AppColors.emotionDisgust;
    case 'fear':
      return AppColors.emotionFear;
    case 'happy':
      return AppColors.emotionHappy;
    case 'neutral':
      return AppColors.emotionNeutral;
    case 'sad':
      return AppColors.emotionSad;
    case 'surprise':
      return AppColors.emotionSurprise;
    default:
      return AppColors.primary;
  }
}
