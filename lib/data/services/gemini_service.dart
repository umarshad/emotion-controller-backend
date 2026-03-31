import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart';
import '../config/api_config.dart';
import '../models/emotion_model.dart';
import '../utils/mock_data.dart';

class GeminiService {
  static final GeminiService _instance = GeminiService._internal();

  factory GeminiService() {
    return _instance;
  }

  GeminiService._internal();

  /// Analyze emotion and generate response using Gemini
  /// Returns a Map with 'emotion' (EmotionModel or null) and 'response' (String)
  Future<Map<String, dynamic>> analyzeEmotionAndChat(String userText) async {
    try {
      final apiKey = ApiConfig.geminiKey;
      if (apiKey.isEmpty) {
        throw Exception('Gemini API key is missing');
      }

      final model = ApiConfig.geminiModel;
      final url = Uri.parse(
        'https://generativelanguage.googleapis.com/v1beta/models/$model:generateContent?key=$apiKey',
      );

      // System prompt to guide the AI
      const systemPrompt = '''
You are an empathetic emotional support assistant. Your goal is to:
1. Analyze the EMOTION in the user's text.
2. Provide a supportive, natural, conversational RESPONSE (max 2-3 sentences).
3. Generate a reflective JOURNALLING PROMPT base on the emotion to help the user explore their feelings.
4. Return the result strictly as a JSON object.

JSON Format:
{
  "emotion": "happy" | "sad" | "angry" | "surprise" | "fear" | "neutral" | "disgust",
  "confidence": <number 0-100>,
  "response": "<your conversational response>",
  "journalling_prompt": "<reflective prompt for the user>"
}

Example User: "I finally got the job!"
Example Output:
{
  "emotion": "happy",
  "confidence": 95,
  "response": "That is absolutely wonderful news! I am so thrilled for you, you worked so hard for this.",
  "journalling_prompt": "How does this achievement align with your long-term goals?"
}
''';

      debugPrint('[Gemini] 🔑 Using API Key: ${apiKey.substring(0, 8)}...');
      debugPrint('[Gemini] 🤖 Model: $model');

      final response = await http.post(
        url,
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode({
          'contents': [
            {
              'parts': [
                {'text': 'System Instruction: $systemPrompt\n\nUser Input: $userText'}
              ]
            }
          ],
          'generationConfig': {
            'temperature': 0.7,
            'topK': 40,
            'topP': 0.95,
            'maxOutputTokens': 1024,
            'responseMimeType': 'application/json',
          },
        }),
      );

      debugPrint('[Gemini] Response Status: ${response.statusCode}');
      // debugPrint('[Gemini] Response Body: ${response.body}');

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final content = data['candidates'][0]['content']['parts'][0]['text'];

        // Parse the JSON content from the AI
        debugPrint('[Gemini] Raw Content: $content');
        try {
          final jsonContent = jsonDecode(content);
          final emotionType =
              jsonContent['emotion']?.toString().toLowerCase() ?? 'neutral';
          final confidence = (jsonContent['confidence'] as num?)?.toInt() ?? 50;
          final chatResponse =
              jsonContent['response']?.toString() ?? "I hear you.";
          final journallingPrompt = jsonContent['journalling_prompt']?.toString();

          // Validate emotion type
          String validatedType = 'neutral';
          const validTypes = [
            'happy',
            'sad',
            'angry',
            'surprise',
            'fear',
            'neutral',
            'disgust',
          ];
          if (validTypes.contains(emotionType)) {
            validatedType = emotionType;
          }

          final emotion = EmotionModel(
            id: DateTime.now().toIso8601String(),
            type: validatedType,
            intensity: confidence,
            timestamp: DateTime.now(),
            detectionMethod: 'chat',
            suggestions: MockEmotionService.getSuggestionsForEmotion(
              validatedType,
            ),
            journallingPrompt: journallingPrompt,
          );

          return {'emotion': emotion, 'response': chatResponse};
        } catch (e) {
          debugPrint('[Gemini] Failed to parse content JSON: $e');
          return {
            'emotion': null,
            'response': content,
          };
        }
      } else {
        final errorBody = response.body;
        debugPrint('[Gemini] API Error: ${response.statusCode}');
        debugPrint('[Gemini] Response Body: $errorBody');

        if (response.statusCode == 401 || response.statusCode == 403) {
          throw Exception(
            'Gemini API Authentication failed: Please check if your API key is valid.',
          );
        } else if (response.statusCode == 404) {
          throw Exception(
            'Gemini Model not found: The model "$model" might be invalid.',
          );
        } else if (response.statusCode == 429) {
          throw Exception(
            'Gemini Rate limit exceeded: Please try again later.',
          );
        }

        throw Exception(
          'Failed to connect to Gemini: ${response.statusCode} - $errorBody',
        );
      }
    } on http.ClientException catch (e) {
      debugPrint('[Gemini] Client Exception: $e');
      throw Exception('Network error: Please check your internet connection.');
    } catch (e) {
      debugPrint('[Gemini] Unexpected Exception: $e');
      rethrow;
    }
  }
}
