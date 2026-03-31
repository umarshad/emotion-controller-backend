import 'package:get/get.dart';
import '../models/emotion_model.dart';
import '../models/emotion_model.dart' as models;
// import '../utils/mock_data.dart'; // Keep for initial greeting or fallbacks if needed
import '../services/gemini_service.dart';
import 'emotion_controller.dart';

class ChatController extends GetxController {
  final RxList<models.ChatMessage> messages = <models.ChatMessage>[].obs;
  final RxString inputText = ''.obs;
  final RxBool isTyping = false.obs;
  final Rx<EmotionModel?> lastDetectedEmotion = Rx<EmotionModel?>(null);
  final RxBool isLoading = false.obs;

  @override
  void onInit() {
    super.onInit();
    _addInitialGreeting();
  }

  void _addInitialGreeting() {
    messages.add(
      models.ChatMessage(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        text:
            'Hello! I\'m here to help you understand your emotions. Tell me how you\'re feeling today.',
        isUser: false,
        timestamp: DateTime.now(),
      ),
    );
  }

  /// Send a message
  Future<void> sendMessage(String text) async {
    if (text.trim().isEmpty) return;

    // Add user message
    final userMessage = models.ChatMessage(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      text: text.trim(),
      isUser: true,
      timestamp: DateTime.now(),
    );
    messages.add(userMessage);
    inputText.value = '';

    // Show typing indicator
    isTyping.value = true;
    isLoading.value = true;

    try {
      // Use Gemini Service
      final result = await GeminiService().analyzeEmotionAndChat(text);

      final detectedEmotion = result['emotion'] as models.EmotionModel?;
      final aiResponse = result['response'] as String;

      // Add AI response message
      final aiMessage = models.ChatMessage(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        text: aiResponse,
        isUser: false,
        timestamp: DateTime.now(),
        detectedEmotion: detectedEmotion,
      );
      messages.add(aiMessage);

      // Update last detected emotion
      if (detectedEmotion != null) {
        lastDetectedEmotion.value = detectedEmotion;
        // Add to emotion controller
        Get.find<EmotionController>().addEmotion(detectedEmotion);
      }
    } catch (e) {
      print('[ChatController] Error: $e');

      // Extract clean error message if it starts with 'Exception: '
      String errorMessage = e.toString();
      if (errorMessage.startsWith('Exception: ')) {
        errorMessage = errorMessage.replaceFirst('Exception: ', '');
      }

      // Add AI response message with error info
      messages.add(
        models.ChatMessage(
          id: DateTime.now().millisecondsSinceEpoch.toString(),
          text: "Error: $errorMessage",
          isUser: false,
          timestamp: DateTime.now(),
        ),
      );
    } finally {
      // Hide typing indicator
      isTyping.value = false;
      isLoading.value = false;
    }
  }

  /// Clear chat history
  void clearChat() {
    messages.clear();
    lastDetectedEmotion.value = null;
    _addInitialGreeting();
  }

  /// Update input text
  void updateInputText(String text) {
    inputText.value = text;
  }
}
