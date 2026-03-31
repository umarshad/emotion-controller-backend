import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import '../../data/constants/colors.dart';
import '../../data/constants/strings.dart';
import '../../data/controllers/chat_controller.dart';
import '../../data/controllers/theme_controller.dart';
import '../../widgets/custom_text.dart';
import '../../widgets/custom_app_bar.dart';
import '../../widgets/emotion_rating_dialog.dart';
import '../../data/controllers/emotion_controller.dart';
import 'widgets/chat_bubble.dart';
import 'widgets/emotion_response_widget.dart';

/// AI Chat screen for text-based emotion detection
class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  late TextEditingController _textController;
  late ScrollController _scrollController;
  final ChatController _chatController = Get.find<ChatController>();

  @override
  void initState() {
    super.initState();
    _textController = TextEditingController();
    _scrollController = ScrollController();

    // Listen to messages changes to auto-scroll
    ever(_chatController.messages, (_) {
      _scrollToBottom();
    });
  }

  @override
  void dispose() {
    _textController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final chatController = _chatController;
    final themeController = Get.find<ThemeController>();

    return Obx(() {
      // Access observable to make this reactive
      final _ = themeController.themeMode.value;
      return Scaffold(
        backgroundColor: Theme.of(context).scaffoldBackgroundColor,
        appBar: CustomAppBar(
          title: AppStrings.chatTitle,
          showBackButton: false,
        ),
        body: SafeArea(
          bottom: false, // Bottom safe area handled by nav bar
          child: Column(
            children: [
              // Messages list
              Expanded(
                child: Obx(() {
                  if (chatController.messages.isEmpty) {
                    return Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            Icons.chat_bubble_outline,
                            size: 64.sp,
                            color: AppColors.textLight,
                          ),
                          SizedBox(height: 16.h),
                          CustomText(
                            AppStrings.chatEmpty,
                            fontSize: 16,
                            color: AppColors.textSecondary,
                          ),
                        ],
                      ),
                    );
                  }

                  return ListView.builder(
                    controller: _scrollController,
                    padding: EdgeInsets.symmetric(vertical: 8.h),
                    itemCount:
                        chatController.messages.length +
                        (chatController.isTyping.value ? 1 : 0),
                    itemBuilder: (context, index) {
                      if (index == chatController.messages.length &&
                          chatController.isTyping.value) {
                        return _buildTypingIndicator();
                      }

                      final message = chatController.messages[index];
                      final isLastMessage =
                          index == chatController.messages.length - 1;
                      final isLastAIResponse = !message.isUser && isLastMessage;

                      return Column(
                        children: [
                          ChatBubble(message: message),
                          // Show emotion response after AI message with detected emotion
                          if (isLastAIResponse &&
                              message.detectedEmotion != null) ...[
                            EmotionResponseWidget(
                              emotion: message.detectedEmotion!,
                            ),
                            SizedBox(height: 8.h),
                            // Reactive Rating Display
                            Obx(() {
                              final emotionController =
                                  Get.find<EmotionController>();
                              final currentEmotionState = emotionController
                                  .emotions
                                  .firstWhereOrNull(
                                      (e) => e.id == message.detectedEmotion!.id);

                              final displayRating =
                                  currentEmotionState?.userRating ??
                                      message.detectedEmotion!.userRating;

                              return Padding(
                                padding: EdgeInsets.symmetric(horizontal: 16.w),
                                child: InkWell(
                                  onTap: () {
                                    Get.dialog(
                                      EmotionRatingDialog(
                                        emotionId: message.detectedEmotion!.id,
                                        initialRating: displayRating ?? 0,
                                        onRatingSelected: (rating) {
                                          emotionController.updateEmotionRating(
                                            message.detectedEmotion!.id,
                                            rating,
                                          );
                                        },
                                      ),
                                    );
                                  },
                                  borderRadius: BorderRadius.circular(12.r),
                                  child: Container(
                                    padding: EdgeInsets.symmetric(
                                      horizontal: 12.w,
                                      vertical: 6.h,
                                    ),
                                    decoration: BoxDecoration(
                                      color:
                                          AppColors.primary.withValues(alpha: 0.05),
                                      borderRadius: BorderRadius.circular(20.r),
                                      border: Border.all(
                                        color: AppColors.primary
                                            .withValues(alpha: 0.1),
                                      ),
                                    ),
                                    child: Row(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        if (displayRating != null &&
                                            displayRating > 0)
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
                                              size: 18.sp,
                                            ),
                                          )
                                        else
                                          CustomText(
                                            'Rate Now',
                                            fontSize: 12,
                                            fontWeight: FontWeight.w600,
                                            color: AppColors.primary,
                                          ),
                                        SizedBox(width: 4.w),
                                        Icon(
                                          Icons.edit,
                                          size: 12.sp,
                                          color: AppColors.primary,
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              );
                            }),
                          ],
                        ],
                      );
                    },
                  );
                }),
              ),

              // Input field
              Container(
                padding: EdgeInsets.only(
                  left: 16.w,
                  right: 16.w,
                  top: 16.h,
                  bottom: 16.h,
                ),
                decoration: BoxDecoration(
                  color: AppColors.surface,
                  boxShadow: [
                    BoxShadow(
                      color: AppColors.shadow,
                      blurRadius: 4,
                      offset: const Offset(0, -2),
                    ),
                  ],
                ),
                child: SafeArea(
                  top: false,
                  child: Row(
                    children: [
                      Expanded(
                        child: Container(
                          decoration: BoxDecoration(
                            color: AppColors.cardBackground,
                            borderRadius: BorderRadius.circular(24.r),
                            border: Border.all(color: AppColors.border),
                          ),
                          child: TextField(
                            controller: _textController,
                            decoration: InputDecoration(
                              hintText: AppStrings.chatHint,
                              hintStyle: TextStyle(
                                color: AppColors.textLight,
                                fontSize: 14.sp,
                              ),
                              border: InputBorder.none,
                              contentPadding: EdgeInsets.symmetric(
                                horizontal: 16.w,
                                vertical: 12.h,
                              ),
                            ),
                            style: TextStyle(fontSize: 14.sp),
                            maxLines: null,
                            textInputAction: TextInputAction.send,
                            onSubmitted: (text) {
                              if (text.trim().isNotEmpty) {
                                chatController.sendMessage(text);
                                _textController.clear();
                              }
                            },
                          ),
                        ),
                      ),
                      SizedBox(width: 8.w),
                      Obx(
                        () => IconButton(
                          onPressed: chatController.isLoading.value
                              ? null
                              : () {
                                  if (_textController.text.trim().isNotEmpty) {
                                    chatController.sendMessage(
                                      _textController.text,
                                    );
                                    _textController.clear();
                                  }
                                },
                          icon: Icon(
                            Icons.send,
                            color: chatController.isLoading.value
                                ? AppColors.textLight
                                : AppColors.primary,
                          ),
                          style: IconButton.styleFrom(
                            backgroundColor: AppColors.primary.withValues(
                              alpha: 0.1,
                            ),
                            padding: EdgeInsets.all(12.w),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      );
    });
  }

  Widget _buildTypingIndicator() {
    return Container(
      margin: EdgeInsets.only(top: 8.h, bottom: 8.h, left: 16.w, right: 40.w),
      padding: EdgeInsets.symmetric(horizontal: 16.w, vertical: 12.h),
      decoration: BoxDecoration(
        color: AppColors.cardBackground,
        borderRadius: BorderRadius.circular(16.r),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          _buildTypingDot(0),
          SizedBox(width: 4.w),
          _buildTypingDot(1),
          SizedBox(width: 4.w),
          _buildTypingDot(2),
        ],
      ),
    );
  }

  Widget _buildTypingDot(int index) {
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0.0, end: 1.0),
      duration: const Duration(milliseconds: 600),
      onEnd: () {
        // Restart animation
      },
      builder: (context, value, child) {
        final delay = index * 0.2;
        final animatedValue = ((value + delay) % 1.0);
        return Container(
          width: 8.w,
          height: 8.h,
          decoration: BoxDecoration(
            color: AppColors.textSecondary.withValues(
              alpha: 0.3 + (animatedValue * 0.7),
            ),
            shape: BoxShape.circle,
          ),
        );
      },
    );
  }
}
