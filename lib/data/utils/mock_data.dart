import 'dart:math';
import 'dart:ui';
import '../models/emotion_model.dart';

/// Mock emotion service for simulating backend behavior (frontend only)
/// Designed to be easily swapped with real API service later
class MockEmotionService {
  static final Random _random = Random();

  /// Detect emotion from camera with random selection and 60-95% intensity
  /// Simulates live camera detection with realistic timing
  /// Uses only the 7 backend-trained emotions
  /// [previousEmotionType] - If provided, has 35% probability of returning the same emotion (reduced stickiness for real-time testing)
  static Future<EmotionModel> detectEmotionFromCamera({
    String? previousEmotionType,
  }) async {
    // Use only 7 backend emotions
    const emotions = [
      'angry',
      'disgust',
      'fear',
      'happy',
      'neutral',
      'sad',
      'surprise',
    ];
    String selectedEmotion;

    // Add stickiness: if previous emotion exists, 35% chance to keep it (reduced from 75% for more variety)
    if (previousEmotionType != null &&
        emotions.contains(previousEmotionType.toLowerCase())) {
      final shouldKeepSame =
          _random.nextInt(100) < 35; // 35% probability (reduced from 75%)
      if (shouldKeepSame) {
        selectedEmotion = previousEmotionType.toLowerCase();
      } else {
        // 65% chance to change to a different emotion (increased from 25%)
        final otherEmotions = emotions
            .where((e) => e != previousEmotionType.toLowerCase())
            .toList();
        final randomIndex = _random.nextInt(otherEmotions.length);
        selectedEmotion = otherEmotions[randomIndex];
      }
    } else {
      // No previous emotion, select randomly
      final randomIndex = _random.nextInt(emotions.length);
      selectedEmotion = emotions[randomIndex];
    }

    // Generate intensity in 60-95% range for camera
    final intensity = 60 + _random.nextInt(36); // 60 to 95

    return _createMockEmotion(selectedEmotion, 'camera', intensity);
  }

  /// Detect emotion from text with enhanced keyword matching, negation detection, and scoring
  /// Returns null if no emotion should be detected, otherwise returns emotion with appropriate intensity
  static Future<EmotionModel?> detectEmotionFromText(String text) async {
    final lowerText = text.toLowerCase().trim();

    if (lowerText.isEmpty) {
      return null;
    }

    // Expanded keyword matching for 7 backend emotions with intensity tiers
    final emotionKeywords = {
      'angry': [
        // Intense (90-100% intensity)
        'furious', 'enraged', 'livid', 'irate', 'outraged',
        // Strong (70-90%)
        'angry', 'mad', 'irritated', 'pissed', 'infuriated',
        // Moderate (50-70%)
        'annoyed', 'frustrated', 'upset', 'bothered', 'agitated',
        // Mild (30-50%)
        'displeased', 'irked', 'cross', 'ticked',
        // Related terms
        'rage', 'anger', 'resentment', 'temper', 'wrath', 'hostile',
      ],
      'happy': [
        // Intense
        'ecstatic', 'overjoyed', 'thrilled', 'elated', 'euphoric',
        // Strong
        'happy', 'joyful', 'delighted', 'cheerful', 'excited',
        // Moderate
        'glad', 'pleased', 'content', 'satisfied', 'grateful',
        // Mild
        'good', 'fine', 'okay', 'alright', 'nice',
        // Related terms
        'joy',
        'happiness',
        'bliss',
        'delight',
        'wonderful',
        'amazing',
        'great',
        'fantastic',
      ],
      'sad': [
        // Intense
        'devastated', 'heartbroken', 'miserable', 'despairing', 'hopeless',
        // Strong
        'sad', 'depressed', 'unhappy', 'sorrowful', 'gloomy',
        // Moderate
        'down', 'melancholy', 'blue', 'upset', 'disappointed',
        // Mild
        'somber', 'dejected', 'low', 'bummed',
        // Related terms
        'sadness', 'grief', 'sorrow', 'despair', 'lonely', 'alone', 'crying',
      ],
      'fear': [
        // Intense
        'terrified', 'petrified', 'horrified', 'panicked', 'traumatized',
        // Strong
        'afraid', 'scared', 'frightened', 'fearful', 'anxious',
        // Moderate
        'worried', 'nervous', 'uneasy', 'apprehensive', 'concerned',
        // Mild
        'tense', 'jittery', 'hesitant', 'uncertain',
        // Related terms
        'fear', 'anxiety', 'panic', 'dread', 'terror', 'stress', 'stressed',
      ],
      'disgust': [
        // Intense
        'revolted', 'repulsed', 'sickened', 'appalled', 'nauseated',
        // Strong
        'disgusted', 'disgusting', 'gross', 'revolting', 'repugnant',
        // Moderate
        'distaste', 'dislike', 'offended', 'repelled',
        // Mild
        'unpleasant', 'disagreeable', 'distasteful',
        // Related terms
        'disgust', 'revulsion', 'nausea', 'loathing', 'yucky',
      ],
      'surprise': [
        // Intense
        'shocked', 'astounded', 'stunned', 'astonished', 'flabbergasted',
        // Strong
        'surprised', 'amazed', 'startled', 'bewildered', 'dumbfounded',
        // Moderate
        'unexpected', 'sudden', 'puzzled', 'confused', 'baffled',
        // Mild
        'curious', 'wondering', 'uncertain',
        // Related terms
        'surprise', 'shock', 'wow', 'whoa', 'unexpected',
      ],
      'neutral': [
        // All moderate
        'neutral', 'calm', 'peaceful', 'serene', 'tranquil',
        'quiet', 'okay', 'fine', 'balanced', 'steady',
        'composed', 'stable', 'relaxed', 'chill', 'meh',
      ],
    };

    // Negation words to detect
    final negations = [
      'not',
      'never',
      'no',
      "don't",
      "doesn't",
      "didn't",
      "won't",
      "wouldn't",
      "can't",
      "couldn't",
      "shouldn't",
      "isn't",
      "aren't",
      "wasn't",
      "weren't",
      "haven't",
      "hasn't",
      'rather than',
      'instead of',
      'anything but',
      'far from',
    ];

    // Helper function to check if keyword is negated
    bool isNegated(String keyword) {
      final keywordIndex = lowerText.indexOf(keyword.toLowerCase());
      if (keywordIndex == -1) return false;

      // Check 5 words (approx 30 characters) before keyword for negation
      final startIndex = (keywordIndex - 30).clamp(0, lowerText.length);
      final beforeKeyword = lowerText.substring(startIndex, keywordIndex);

      for (final negation in negations) {
        if (beforeKeyword.contains(negation)) {
          return true;
        }
      }

      return false;
    }

    // Score map for all emotions
    Map<String, int> emotionScores = {
      'angry': 0,
      'disgust': 0,
      'fear': 0,
      'happy': 0,
      'neutral': 0,
      'sad': 0,
      'surprise': 0,
    };

    // Find matching keywords with negation check
    for (final entry in emotionKeywords.entries) {
      for (int i = 0; i < entry.value.length; i++) {
        final keyword = entry.value[i];
        if (lowerText.contains(keyword)) {
          if (isNegated(keyword)) {
            // Negation detected - boost opposite emotions
            if (entry.key == 'happy') {
              emotionScores['sad'] = (emotionScores['sad']! + 12);
              emotionScores['neutral'] = (emotionScores['neutral']! + 8);
            } else if (entry.key == 'sad') {
              emotionScores['happy'] = (emotionScores['happy']! + 12);
              emotionScores['neutral'] = (emotionScores['neutral']! + 8);
            } else if (entry.key == 'angry') {
              emotionScores['calm'] = (emotionScores['neutral']! + 10);
            } else {
              // For other emotions, boost neutral
              emotionScores['neutral'] = (emotionScores['neutral']! + 8);
            }
          } else {
            // Normal keyword match - score based on position (intensity tier)
            int score = 10; // Base score

            // Intense keywords (first 5) = highest score
            if (i < 5) {
              score = 15;
            }
            // Strong keywords (6-10) = high score
            else if (i < 10) {
              score = 13;
            }
            // Moderate keywords (11-14) = medium score
            else if (i < 14) {
              score = 11;
            }
            // Mild keywords and related terms = lower score
            else {
              score = 8;
            }

            emotionScores[entry.key] = emotionScores[entry.key]! + score;
          }
        }
      }
    }

    // Find emotion with highest score
    final sortedEmotions = emotionScores.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    final topEmotion = sortedEmotions.first;

    // If score is too low (ambiguous message), return neutral instead of random
    if (topEmotion.value < 5) {
      return _createMockEmotion('neutral', 'chat', 30);
    }

    // Calculate intensity based on score (higher score = higher intensity)
    final intensity = (topEmotion.value * 4).clamp(30, 95);

    return _createMockEmotion(topEmotion.key, 'chat', intensity);
  }

  /// Extract key topics/subjects from user message for personalized responses
  static List<String> _extractKeyTopics(String text) {
    final lowerText = text.toLowerCase();
    final topics = <String>[];

    // Common topic patterns for emotional content
    final topicPatterns = {
      r'\b(work|job|boss|colleague|office|career|workplace|employee|employer)\b':
          'work',
      r'\b(family|parent|mom|mother|dad|father|sibling|brother|sister|relative)\b':
          'family',
      r'\b(friend|friends|friendship|buddy|pal)\b': 'friendship',
      r'\b(relationship|partner|boyfriend|girlfriend|spouse|marriage|dating)\b':
          'relationship',
      r'\b(school|class|teacher|professor|exam|homework|study|student|college|university)\b':
          'school',
      r'\b(health|sick|illness|pain|doctor|hospital|medical)\b': 'health',
      r'\b(money|financial|debt|bills|income|salary|broke|expensive)\b':
          'finances',
      r'\b(life|living|situation|day|week|time|moment)\b': 'life',
    };

    for (final pattern in topicPatterns.entries) {
      if (RegExp(pattern.key).hasMatch(lowerText)) {
        topics.add(pattern.value);
      }
    }

    return topics;
  }

  /// Simulate backend delay for realistic API response timing
  /// Chat: 1-2 seconds random
  /// Camera: variable (handled by timer in controller)
  static Future<void> simulateBackendDelay({
    Duration? customDelay,
    bool isChat = true,
  }) async {
    if (customDelay != null) {
      await Future.delayed(customDelay);
      return;
    }

    if (isChat) {
      // Random delay between 1-2 seconds for chat
      final delayMs = 1000 + _random.nextInt(1000); // 1000-2000ms
      await Future.delayed(Duration(milliseconds: delayMs));
    } else {
      // Camera delays are handled by timer intervals (2-4 seconds)
      await Future.delayed(const Duration(milliseconds: 100));
    }
  }

  /// Create mock emotion with suggestions
  /// For camera detections, includes face detection data (faceDetected: true, mock faceBounds)
  static EmotionModel _createMockEmotion(
    String type,
    String method, [
    int? customIntensity,
  ]) {
    final intensity = customIntensity ?? _generateIntensity(type);
    final suggestions = getSuggestionsForEmotion(type);

    // For camera detections, include face detection data so overlay works
    if (method == 'camera') {
      // Create mock face bounds (centered box, typical face size)
      // These are in image coordinate space (will be converted to screen coordinates in UI)
      // Typical camera resolution: 720x480 or similar
      // Face typically occupies ~20-30% of frame width/height
      const mockImageWidth = 720.0;
      const mockImageHeight = 480.0;
      const faceWidth = 200.0; // ~28% of width
      const faceHeight = 250.0; // ~52% of height
      const faceX = (mockImageWidth - faceWidth) / 2; // Center horizontally
      const faceY = (mockImageHeight - faceHeight) / 2; // Center vertically

      return EmotionModel(
        id: DateTime.now().toIso8601String(),
        type: type,
        intensity: intensity,
        timestamp: DateTime.now(),
        detectionMethod: method,
        suggestions: suggestions,
        journallingPrompt: getJournallingPromptForEmotion(type),
        faceDetected: true, // Mock service simulates face detection
        faceBounds: Rect.fromLTWH(faceX, faceY, faceWidth, faceHeight),
        numFaces: 1,
      );
    }

    // For chat detections, no face detection data
    return EmotionModel(
      id: DateTime.now().toIso8601String(),
      type: type,
      intensity: intensity,
      timestamp: DateTime.now(),
      detectionMethod: method,
      suggestions: suggestions,
      journallingPrompt: getJournallingPromptForEmotion(type),
      faceDetected: false,
      faceBounds: null,
      numFaces: 0,
    );
  }

  /// Generate intensity based on emotion type (with variation)
  /// Uses only 7 backend emotions
  static int _generateIntensity(String type) {
    // Simulate realistic intensity ranges for 7 backend emotions
    final baseIntensities = {
      'angry': 65,
      'disgust': 60,
      'fear': 50,
      'happy': 75,
      'neutral': 30,
      'sad': 55,
      'surprise': 45,
    };

    final base = baseIntensities[type.toLowerCase()] ?? 50;
    // Add random variation (-15 to +15)
    final variation = _random.nextInt(31) - 15;
    return (base + variation).clamp(10, 100);
  }

  /// Get personalized suggestions for each emotion
  /// Made public for use by API service
  /// Uses only 7 backend emotions
  static List<String> getSuggestionsForEmotion(String type) {
    final suggestionsMap = {
      'angry': [
        'Practice a 5-minute guided mindfulness meditation focusing on "releasing"',
        'Engage in a vigorous physical activity to channel the intense energy',
        'Try a "brain dump" - write down every single frustration without filter',
        'Use the "stop and breathe" technique: inhale for 4, hold for 4, exhale for 8',
        'Physically move to a different room or environment to break the cycle',
      ],
      'disgust': [
        'Identify the exact sensory or moral trigger and acknowledge it',
        'Visualize a clear, protective boundary between yourself and the trigger',
        'Focus on a "refreshing" activity: wash your face or tidy a small area',
        'Acknowledge your high standards and reflect on what they protect',
        'Practice self-compassion if the feeling is directed inward',
      ],
      'fear': [
        'Use the "5-4-3-2-1" grounding technique to return to the present moment',
        'Identify one small, concrete action you can take to feel safer',
        'Verbally label the fear - "I am feeling afraid because..." - to reduce its power',
        'Reach out to a trusted "safe person" for a 5-minute grounding chat',
        'Repeat a calming mantra: "I am safe in this moment, and this will pass"',
      ],
      'happy': [
        'Journal about this specific moment of joy to "anchor" it in your memory',
        'Express your gratitude to someone who contributed to this positive state',
        'Savor the feeling by describing it in 3 words to yourself',
        'Do something small and kind for another person to amplify the joy',
        'Listen to a piece of music that matches the rhythm of your happiness',
      ],
      'neutral': [
        'Take a moment to appreciate the quiet power of emotional equilibrium',
        'Engage in a low-stakes creative activity like sketching or organizing',
        'Set a simple intention for your next activity while in this clear state',
        'Practice mindful listening - notice every sound in your current environment',
        'Observe your breath without trying to change it, simply noticing the flow',
      ],
      'sad': [
        'Allow yourself to feel the weight of the emotion without trying to "fix" it',
        'Write a letter to your sadness, asking what it needs you to know',
        'Engage in gentle movement - a slow walk or light stretching in soft light',
        'Create a "comfort nest" with a warm blanket and a soothing beverage',
        'Reach out to a friend with a simple: "I am having a hard time, can we talk?"',
      ],
      'surprise': [
        'Take three slow, deep breaths to integrate the unexpected information',
        'Pause before taking any action - let the initial shock settle first',
        'Ask: "What is the most interesting thing about this unexpected event?"',
        'Draw or map out what happened to see it from a logical perspective',
        'Discuss the surprise with someone whose judgment you trust',
      ],
    };

    return suggestionsMap[type.toLowerCase()] ??
        [
          'Take care of yourself and listen to your needs',
          'Be kind and patient with yourself in this moment',
          'Seek support from those you trust if you need it',
        ];
  }

  /// Get a default journalling prompt for an emotion
  static String getJournallingPromptForEmotion(String type) {
    final prompts = {
      'angry': 'What specifically triggered your anger in this moment?',
      'disgust': 'What part of this situation feels most off-putting to you?',
      'fear': 'What is the smallest step you could take to feel a bit safer?',
      'happy': 'What are you most grateful for in this positive moment?',
      'neutral':
          'How does this moment of calm feel compared to more intense times?',
      'sad': 'If your sadness had a message for you, what would it be?',
      'surprise': 'What was the most unexpected part of this event?',
    };

    return prompts[type.toLowerCase()] ??
        'How are you feeling about this moment?';
  }

  /// Generate AI chat response based on detected emotion with personalization
  /// Uses topic extraction and multiple templates for variety and relevance
  static String generateAIResponse(String emotionType, String userMessage) {
    // Extract key topics from user message for personalization
    final topics = _extractKeyTopics(userMessage);
    final topicMention = topics.isNotEmpty ? topics.first : null;

    // Multiple response templates per emotion for variety
    final responseTemplates = {
      'angry': [
        topicMention != null
            ? "I can sense frustration about {topic}. It's completely natural to feel this way. Let's work through this together."
            : "I can sense some frustration in your message. It's completely normal to feel this way. Let's work through this together.",
        topicMention != null
            ? "Your anger about {topic} is valid. Sometimes things can be overwhelming. I'm here to support you."
            : "Your anger is valid. Sometimes things can be overwhelming. I'm here to support you.",
        "I hear the frustration in your words. Feeling angry is okay - it shows you care. Let's explore what might help.",
      ],
      'happy': [
        topicMention != null
            ? "It's wonderful to hear that you're feeling happy about {topic}! I'm glad you're experiencing positive emotions."
            : "It's wonderful to hear that you're feeling happy! I'm glad you're experiencing positive emotions today.",
        "Your happiness is contagious! It's great to see you in such a positive state. Keep savoring these moments!",
        topicMention != null
            ? "I love hearing about your joy with {topic}! These positive feelings are important - embrace them fully."
            : "I love hearing about your joy! These positive feelings are important - embrace them fully.",
      ],
      'sad': [
        topicMention != null
            ? "I hear the sadness about {topic}. Your feelings are completely valid, and I'm here to support you through this."
            : "I hear the sadness in your words. Your feelings are valid, and I'm here to support you through this.",
        topicMention != null
            ? "It's okay to feel sad about {topic}. These emotions are natural. You don't have to face this alone."
            : "It's okay to feel sad. These emotions are part of being human. You don't have to face this alone.",
        "I'm sorry you're going through this difficult time. Sadness can be heavy, but it's temporary. I'm here with you.",
      ],
      'fear': [
        topicMention != null
            ? "Fear about {topic} is completely understandable. Let's explore what might help you feel more secure."
            : "Fear is a natural response. Let's explore what might help you feel more secure.",
        topicMention != null
            ? "I can sense your anxiety about {topic}. Your concerns are valid - they matter. Let's work through this together."
            : "I can sense your anxiety. Fear often signals that something matters to us. Let's work through this together.",
        "It's okay to feel afraid. Acknowledging your fear is brave. I'm here to help you navigate this.",
      ],
      'neutral': [
        "I can sense a balanced state in your message. Maintaining equilibrium can be beneficial. How are you feeling about things?",
        topicMention != null
            ? "It sounds like you're in a calm space regarding {topic}. That's good - how can I support you today?"
            : "It sounds like you're in a calm space right now. That's good - how can I support you today?",
        "I'm picking up a neutral tone. Sometimes that's the most comfortable place to be. What's on your mind?",
      ],
      'disgust': [
        topicMention != null
            ? "I understand you're feeling disgusted about {topic}. These feelings are valid, and I'm here to support you."
            : "I understand that you're feeling disgusted. These feelings are valid, and I'm here to support you.",
        "Your disgust is a natural reaction. It shows you have strong values. Let's talk about what's bothering you.",
        topicMention != null
            ? "I can sense your strong reaction to {topic}. These feelings matter - let's explore them together."
            : "I can sense your strong reaction. These feelings matter - let's explore them together.",
      ],
      'surprise': [
        topicMention != null
            ? "It sounds like something unexpected happened with {topic}. Surprise can be both exciting and disorienting."
            : "It sounds like something unexpected happened. Surprise can be disorienting, but also an opportunity to learn.",
        "I can sense your surprise! Unexpected events can shake us up, but they also open new perspectives.",
        topicMention != null
            ? "Wow, {topic} caught you off guard! It's natural to feel surprised. Let's talk about it."
            : "Wow, that caught you off guard! It's natural to feel surprised. Let's talk about it.",
      ],
    };

    // Select templates for this emotion
    final templates = responseTemplates[emotionType.toLowerCase()] ?? [];
    if (templates.isEmpty) {
      return 'Thank you for sharing how you\'re feeling. I\'m here to help you understand and manage your emotions.';
    }

    // Select random template for variety
    final randomIndex = _random.nextInt(templates.length);
    var response = templates[randomIndex];

    // Replace {topic} placeholder with actual topic
    if (topicMention != null) {
      response = response.replaceAll('{topic}', topicMention);
    } else {
      // Remove topic-specific phrases if no topic detected
      response = response.replaceAll(' about {topic}', '');
      response = response.replaceAll(' with {topic}', '');
      response = response.replaceAll(' regarding {topic}', '');
      response = response.replaceAll(', especially with {topic}', '');
      response = response.replaceAll(
        ', especially when it comes to {topic}',
        '',
      );
      response = response.replaceAll(' to {topic}', '');
    }

    return response;
  }
}

/// Legacy MockDataService - kept for backward compatibility
/// Use MockEmotionService for new code
@Deprecated('Use MockEmotionService instead')
class MockDataService {
  /// Detect emotion from text (mocked keyword matching)
  /// Note: This is now async to match MockEmotionService
  static Future<EmotionModel?> detectEmotionFromText(String text) async {
    // Use new service
    return await MockEmotionService.detectEmotionFromText(text);
  }

  /// Generate mock emotion from camera (random selection)
  /// Note: This is now async to match MockEmotionService
  static Future<EmotionModel> generateMockCameraEmotion() async {
    // Use new service
    return await MockEmotionService.detectEmotionFromCamera();
  }

  /// Generate AI chat response based on detected emotion
  static String generateAIResponse(String emotionType, String userMessage) {
    return MockEmotionService.generateAIResponse(emotionType, userMessage);
  }
}
