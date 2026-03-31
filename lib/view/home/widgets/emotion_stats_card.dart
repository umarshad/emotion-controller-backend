import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import '../../../data/constants/colors.dart';
import '../../../data/constants/assets.dart';
import '../../../widgets/custom_text.dart';

/// Minimal emotion statistics card
class EmotionStatsCard extends StatelessWidget {
  final int todayCount;
  final int totalCount;
  final String? mostCommonEmotion;

  const EmotionStatsCard({
    super.key,
    required this.todayCount,
    required this.totalCount,
    this.mostCommonEmotion,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.all(20.w),
      decoration: BoxDecoration(
        color: AppColors.cardBackground,
        borderRadius: BorderRadius.circular(16.r),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: AppColors.shadow,
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          CustomText(
            'Today\'s Stats',
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: AppColors.textPrimary,
          ),
          SizedBox(height: 16.h),
          Row(
            children: [
              Expanded(
                child: _buildStatItem(
                  'Today',
                  todayCount.toString(),
                  Icons.today_outlined,
                ),
              ),
              SizedBox(width: 16.w),
              Expanded(
                child: _buildStatItem(
                  'Total',
                  totalCount.toString(),
                  Icons.analytics_outlined,
                ),
              ),
              if (mostCommonEmotion != null) ...[
                SizedBox(width: 16.w),
                Expanded(child: _buildEmotionStat()),
              ],
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildStatItem(String label, String value, IconData icon) {
    return Container(
      padding: EdgeInsets.symmetric(horizontal: 8.w, vertical: 12.h),
      decoration: BoxDecoration(
        color: AppColors.background,
        borderRadius: BorderRadius.circular(12.r),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 20.sp, color: AppColors.primary),
          SizedBox(height: 8.h),
          CustomText(
            value,
            fontSize: 20,
            fontWeight: FontWeight.bold,
            color: AppColors.textPrimary,
          ),
          SizedBox(height: 2.h),
          CustomText(label, fontSize: 12, color: AppColors.textSecondary),
        ],
      ),
    );
  }

  Widget _buildEmotionStat() {
    if (mostCommonEmotion == null) return const SizedBox.shrink();

    final emoji = AppAssets.getEmotionEmoji(mostCommonEmotion!);
    final name = AppAssets.getEmotionName(mostCommonEmotion!);
    final color = _getEmotionColor(mostCommonEmotion!);

    return Container(
      padding: EdgeInsets.symmetric(horizontal: 8.w, vertical: 12.h),
      decoration: BoxDecoration(
        color: AppColors.background,
        borderRadius: BorderRadius.circular(12.r),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(emoji, style: TextStyle(fontSize: 20.sp)),
          SizedBox(height: 8.h),
          CustomText(
            name,
            fontSize: 14,
            fontWeight: FontWeight.w600,
            color: color,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
          SizedBox(height: 2.h),
          CustomText(
            'Most Common',
            fontSize: 12,
            color: AppColors.textSecondary,
          ),
        ],
      ),
    );
  }

  Color _getEmotionColor(String emotionType) {
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
}
