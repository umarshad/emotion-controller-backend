import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import '../../data/constants/colors.dart';
import '../../data/constants/strings.dart';
import '../../data/constants/assets.dart';
import '../../data/controllers/emotion_controller.dart';
import '../../data/controllers/theme_controller.dart';
import '../../data/models/emotion_model.dart';
import '../../data/utils/helpers.dart';
import '../../widgets/custom_text.dart';
import '../../widgets/custom_app_bar.dart';
import 'widgets/emotion_timeline_tile.dart';
import 'widgets/detailed_emotion_view.dart';
import 'widgets/emotion_filter_chip.dart';

/// Enhanced emotion history screen with detailed view and filtering
class EmotionHistoryScreen extends StatelessWidget {
  const EmotionHistoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final emotionController = Get.find<EmotionController>();
    final themeController = Get.find<ThemeController>();
    final RxString selectedMethodFilter = 'all'.obs;
    final Rx<String?> selectedEmotionFilter = Rx<String?>(null);
    final RxString searchQuery = ''.obs;

    return Obx(() {
      // Access observable to make this reactive
      final _ = themeController.themeMode.value;
      return Scaffold(
        backgroundColor: Theme.of(context).scaffoldBackgroundColor,
        appBar: CustomAppBar(
          title: AppStrings.historyTitle,
          showBackButton: false,
          actions: [
            IconButton(
              icon: Icon(
                Icons.search,
                color:
                    Theme.of(context).iconTheme.color ?? AppColors.textPrimary,
              ),
              onPressed: () {
                _showSearchDialog(context, searchQuery);
              },
            ),
          ],
        ),
        body: Column(
          children: [
            // Search bar (if search is active)
            Obx(() {
              if (searchQuery.value.isEmpty) return const SizedBox.shrink();
              return Container(
                padding: EdgeInsets.all(16.w),
                color: Theme.of(context).colorScheme.surface,
                child: Row(
                  children: [
                    Expanded(
                      child: Container(
                        padding: EdgeInsets.symmetric(horizontal: 16.w),
                        decoration: BoxDecoration(
                          color: Theme.of(context).cardColor,
                          borderRadius: BorderRadius.circular(12.r),
                          border: Border.all(
                            color: Theme.of(context).dividerColor,
                          ),
                        ),
                        child: TextField(
                          onChanged: (value) => searchQuery.value = value,
                          decoration: InputDecoration(
                            hintText: 'Search emotions...',
                            hintStyle: TextStyle(
                              color:
                                  Theme.of(
                                    context,
                                  ).textTheme.bodySmall?.color ??
                                  AppColors.textLight,
                              fontSize: 14.sp,
                            ),
                            border: InputBorder.none,
                            icon: Icon(
                              Icons.search,
                              size: 20.sp,
                              color:
                                  Theme.of(
                                    context,
                                  ).textTheme.bodyMedium?.color ??
                                  AppColors.textSecondary,
                            ),
                          ),
                          style: TextStyle(fontSize: 14.sp),
                        ),
                      ),
                    ),
                    SizedBox(width: 8.w),
                    IconButton(
                      icon: Icon(
                        Icons.close,
                        color:
                            Theme.of(context).textTheme.bodyMedium?.color ??
                            AppColors.textSecondary,
                      ),
                      onPressed: () => searchQuery.value = '',
                    ),
                  ],
                ),
              );
            }),

            // Filter chips - Method
            Container(
              padding: EdgeInsets.symmetric(horizontal: 16.w, vertical: 12.h),
              child: Row(
                children: [
                  _buildMethodFilterChip(
                    context,
                    'all',
                    'All',
                    selectedMethodFilter,
                  ),
                  SizedBox(width: 8.w),
                  _buildMethodFilterChip(
                    context,
                    'chat',
                    'Chat',
                    selectedMethodFilter,
                  ),
                  SizedBox(width: 8.w),
                  _buildMethodFilterChip(
                    context,
                    'camera',
                    'Camera',
                    selectedMethodFilter,
                  ),
                ],
              ),
            ),

            // Filter chips - Emotion types (scrollable)
            Container(
              height: 56.h,
              padding: EdgeInsets.symmetric(vertical: 8.h),
              child: ListView(
                scrollDirection: Axis.horizontal,
                padding: EdgeInsets.symmetric(horizontal: 16.w),
                children: [
                  EmotionFilterChip(
                    emotionType: 'all',
                    isSelected: selectedEmotionFilter.value == null,
                    onTap: () => selectedEmotionFilter.value = null,
                  ),
                  SizedBox(width: 8.w),
                  ...AppAssets.allEmotions.map(
                    (emotion) => Padding(
                      padding: EdgeInsets.only(right: 8.w),
                      child: EmotionFilterChip(
                        emotionType: emotion,
                        isSelected: selectedEmotionFilter.value == emotion,
                        onTap: () {
                          final current = selectedEmotionFilter.value;
                          selectedEmotionFilter.value = current == emotion
                              ? null
                              : emotion;
                        },
                      ),
                    ),
                  ),
                ],
              ),
            ),

            // Emotions list
            Expanded(
              child: Obx(() {
                List<EmotionModel> filteredEmotions =
                    emotionController.emotions;

                // Filter by method
                if (selectedMethodFilter.value != 'all') {
                  filteredEmotions = emotionController.getEmotionsByMethod(
                    selectedMethodFilter.value,
                  );
                }

                // Filter by emotion type
                final selectedEmotion = selectedEmotionFilter.value;
                if (selectedEmotion != null && selectedEmotion != 'all') {
                  filteredEmotions = filteredEmotions
                      .where(
                        (e) =>
                            e.type.toLowerCase() ==
                            selectedEmotion.toLowerCase(),
                      )
                      .toList();
                }

                // Filter by search query
                if (searchQuery.value.isNotEmpty) {
                  final query = searchQuery.value.toLowerCase();
                  filteredEmotions = filteredEmotions.where((emotion) {
                    return emotion.type.toLowerCase().contains(query) ||
                        AppAssets.getEmotionName(
                          emotion.type,
                        ).toLowerCase().contains(query);
                  }).toList();
                }

                if (filteredEmotions.isEmpty) {
                  return _buildEmptyState(context);
                }

                // Group by date
                final grouped = Helpers.groupEmotionsByDate(filteredEmotions);
                final sortedDates = grouped.keys.toList()
                  ..sort((a, b) => b.compareTo(a));

                return ListView.builder(
                  padding: EdgeInsets.all(16.w),
                  itemCount: sortedDates.length,
                  itemBuilder: (context, dateIndex) {
                    final date = sortedDates[dateIndex];
                    final emotions = grouped[date]!;
                    final dateTime = DateTime.parse(date);

                    return Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        // Date header
                        Padding(
                          padding: EdgeInsets.only(
                            bottom: 12.h,
                            top: dateIndex > 0 ? 24.h : 0,
                          ),
                          child: CustomText(
                            _getDateHeader(dateTime),
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color:
                                Theme.of(context).textTheme.bodyMedium?.color ??
                                AppColors.textSecondary,
                          ),
                        ),
                        // Emotions for this date
                        ...emotions.asMap().entries.map((entry) {
                          final index = entry.key;
                          final emotion = entry.value;
                          return GestureDetector(
                            onTap: () {
                              Get.to(
                                () => DetailedEmotionView(emotion: emotion),
                              );
                            },
                            child: EmotionTimelineTile(
                              emotion: emotion,
                              isFirst: index == 0,
                              isLast: index == emotions.length - 1,
                            ),
                          );
                        }),
                      ],
                    );
                  },
                );
              }),
            ),
          ],
        ),
      );
    });
  }

  Widget _buildMethodFilterChip(
    BuildContext context,
    String value,
    String label,
    RxString selected,
  ) {
    return Obx(
      () => GestureDetector(
        onTap: () => selected.value = value,
        child: Container(
          padding: EdgeInsets.symmetric(horizontal: 16.w, vertical: 8.h),
          decoration: BoxDecoration(
            color: selected.value == value
                ? AppColors.primary
                : Theme.of(context).cardColor,
            borderRadius: BorderRadius.circular(20.r),
            border: Border.all(
              color: selected.value == value
                  ? AppColors.primary
                  : Theme.of(context).dividerColor,
            ),
          ),
          child: CustomText(
            label,
            fontSize: 14,
            fontWeight: FontWeight.w600,
            color: selected.value == value
                ? Colors.white
                : (Theme.of(context).textTheme.bodyLarge?.color ??
                      AppColors.textPrimary),
          ),
        ),
      ),
    );
  }

  Widget _buildEmptyState(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          TweenAnimationBuilder<double>(
            tween: Tween(begin: 0.0, end: 1.0),
            duration: const Duration(milliseconds: 800),
            builder: (builderContext, value, child) {
              return Transform.scale(
                scale: 0.5 + (value * 0.5),
                child: Opacity(
                  opacity: value,
                  child: Icon(
                    Icons.history,
                    size: 80.sp,
                    color:
                        Theme.of(context).textTheme.bodySmall?.color ??
                        AppColors.textLight,
                  ),
                ),
              );
            },
          ),
          SizedBox(height: 24.h),
          CustomText(
            AppStrings.historyEmpty,
            fontSize: 20,
            fontWeight: FontWeight.w600,
            color:
                Theme.of(context).textTheme.bodyMedium?.color ??
                AppColors.textSecondary,
          ),
          SizedBox(height: 8.h),
          Padding(
            padding: EdgeInsets.symmetric(horizontal: 48.w),
            child: CustomText(
              AppStrings.historyEmptySubtitle,
              fontSize: 14,
              color:
                  Theme.of(context).textTheme.bodySmall?.color ??
                  AppColors.textLight,
              textAlign: TextAlign.center,
            ),
          ),
        ],
      ),
    );
  }

  void _showSearchDialog(BuildContext context, RxString searchQuery) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20.r),
        ),
        title: CustomText(
          'Search Emotions',
          fontSize: 20,
          fontWeight: FontWeight.bold,
        ),
        content: TextField(
          autofocus: true,
          onChanged: (value) => searchQuery.value = value,
          decoration: InputDecoration(
            hintText: 'Type emotion name...',
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(12.r),
            ),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: CustomText('Close', color: AppColors.primary),
          ),
        ],
      ),
    );
  }

  String _getDateHeader(DateTime date) {
    if (Helpers.isToday(date)) {
      return AppStrings.historyToday;
    } else if (Helpers.isYesterday(date)) {
      return AppStrings.historyYesterday;
    } else if (Helpers.isThisWeek(date)) {
      return AppStrings.historyThisWeek;
    } else {
      return Helpers.formatDate(date);
    }
  }
}
