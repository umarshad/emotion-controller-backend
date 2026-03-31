import 'dart:ui';

/// Emotion detection result model
class EmotionModel {
  final String id;
  final String type;
  final int intensity; // 0-100
  final DateTime timestamp;
  final String detectionMethod; // 'chat' or 'camera'
  final List<String> suggestions;
  final String? journallingPrompt;

  final String? imagePath; // Local path to captured image
  final int? userRating; // User rating 1-5 stars
  // ... existing face detection fields ...
  final bool faceDetected; // Whether a face was detected in the frame
  final Rect?
  faceBounds; // Face bounding box in image coordinates (null if no face)
  final int numFaces; // Number of faces detected (0 if no face)

  // Confidence getter (0.0 - 1.0) derived from intensity
  double get confidence => intensity / 100.0;

  EmotionModel({
    required this.id,
    required this.type,
    required this.intensity,
    required this.timestamp,
    required this.detectionMethod,
    required this.suggestions,
    this.journallingPrompt,
    this.imagePath,
    this.userRating,
    this.faceDetected = false,
    this.faceBounds,
    this.numFaces = 0,
  });

  /// Create from JSON (for future API integration)
  factory EmotionModel.fromJson(Map<String, dynamic> json) {
    // Parse face bounds if available
    Rect? faceBounds;
    if (json['faceBounds'] != null) {
      final bounds = json['faceBounds'] as Map<String, dynamic>;
      faceBounds = Rect.fromLTWH(
        (bounds['left'] ?? bounds['x'] ?? 0).toDouble(),
        (bounds['top'] ?? bounds['y'] ?? 0).toDouble(),
        (bounds['width'] ?? 0).toDouble(),
        (bounds['height'] ?? 0).toDouble(),
      );
    }

    return EmotionModel(
      id:
          json['id'] as String? ??
          (json['timestamp'] as String? ??
              DateTime.now()
                  .toIso8601String()), // Fallback to timestamp if id missing
      type: json['type'] as String,
      intensity: json['intensity'] as int,
      timestamp: DateTime.parse(json['timestamp'] as String),
      detectionMethod: json['detectionMethod'] as String,
      suggestions: List<String>.from(json['suggestions'] as List? ?? []),
      journallingPrompt: json['journallingPrompt'] as String?,
      imagePath: json['imagePath'] as String?,
      userRating: json['userRating'] as int?,
      faceDetected: json['faceDetected'] as bool? ?? false,
      faceBounds: faceBounds,
      numFaces: json['numFaces'] as int? ?? 0,
    );
  }

  /// Convert to JSON (for future API integration)
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'type': type,
      'intensity': intensity,
      'timestamp': timestamp.toIso8601String(),
      'detectionMethod': detectionMethod,
      'suggestions': suggestions,
      'journallingPrompt': journallingPrompt,
      'imagePath': imagePath,
      'userRating': userRating,
      'faceDetected': faceDetected,
      'faceBounds': faceBounds != null
          ? {
              'left': faceBounds!.left,
              'top': faceBounds!.top,
              'width': faceBounds!.width,
              'height': faceBounds!.height,
            }
          : null,
      'numFaces': numFaces,
    };
  }

  /// Copy with method for updates
  EmotionModel copyWith({
    String? id,
    String? type,
    int? intensity,
    DateTime? timestamp,
    String? detectionMethod,
    List<String>? suggestions,
    String? journallingPrompt,
    String? imagePath,
    int? userRating,
    bool? faceDetected,
    Rect? faceBounds,
    int? numFaces,
  }) {
    return EmotionModel(
      id: id ?? this.id,
      type: type ?? this.type,
      intensity: intensity ?? this.intensity,
      timestamp: timestamp ?? this.timestamp,
      detectionMethod: detectionMethod ?? this.detectionMethod,
      suggestions: suggestions ?? this.suggestions,
      journallingPrompt: journallingPrompt ?? this.journallingPrompt,
      imagePath: imagePath ?? this.imagePath,
      userRating: userRating ?? this.userRating,
      faceDetected: faceDetected ?? this.faceDetected,
      faceBounds: faceBounds ?? this.faceBounds,
      numFaces: numFaces ?? this.numFaces,
    );
  }
}

/// Chat message model
class ChatMessage {
  final String id;
  final String text;
  final bool isUser;
  final DateTime timestamp;
  final EmotionModel? detectedEmotion;

  ChatMessage({
    required this.id,
    required this.text,
    required this.isUser,
    required this.timestamp,
    this.detectedEmotion,
  });

  ChatMessage copyWith({
    String? id,
    String? text,
    bool? isUser,
    DateTime? timestamp,
    EmotionModel? detectedEmotion,
  }) {
    return ChatMessage(
      id: id ?? this.id,
      text: text ?? this.text,
      isUser: isUser ?? this.isUser,
      timestamp: timestamp ?? this.timestamp,
      detectedEmotion: detectedEmotion ?? this.detectedEmotion,
    );
  }
}
