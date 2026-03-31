import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import '../../../data/constants/colors.dart';
import '../../../data/models/emotion_model.dart' as models;
import '../../../widgets/custom_text.dart';

/// Chat message bubble widget
class ChatBubble extends StatelessWidget {
  final models.ChatMessage message;

  const ChatBubble({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: message.isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: EdgeInsets.only(
          top: 8.h,
          bottom: 8.h,
          left: message.isUser ? 40.w : 16.w,
          right: message.isUser ? 16.w : 40.w,
        ),
        padding: EdgeInsets.symmetric(horizontal: 16.w, vertical: 12.h),
        decoration: BoxDecoration(
          color: message.isUser
              ? AppColors.primary
              : Theme.of(context).cardColor,
          borderRadius: BorderRadius.only(
            topLeft: Radius.circular(16.r),
            topRight: Radius.circular(16.r),
            bottomLeft: message.isUser ? Radius.circular(16.r) : Radius.zero,
            bottomRight: message.isUser ? Radius.zero : Radius.circular(16.r),
          ),
          border: message.isUser
              ? null
              : Border.all(color: Theme.of(context).dividerColor),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            CustomText(
              message.text,
              fontSize: 15,
              color: message.isUser
                  ? Colors.white
                  : (Theme.of(context).textTheme.bodyLarge?.color ??
                        AppColors.textPrimary),
            ),
            if (message.detectedEmotion != null) ...[
              SizedBox(height: 8.h),
              Container(
                padding: EdgeInsets.all(8.w),
                decoration: BoxDecoration(
                  color: message.isUser
                      ? Colors.white.withValues(alpha: 0.2)
                      : AppColors.primary.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(8.r),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      '${message.detectedEmotion!.type} detected',
                      style: TextStyle(
                        fontSize: 12.sp,
                        color: message.isUser
                            ? Colors.white
                            : AppColors.primary,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
