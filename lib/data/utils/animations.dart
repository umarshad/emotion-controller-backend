import 'package:flutter/material.dart';

/// Animation constants and utilities
class AppAnimations {
  // Duration constants
  static const Duration shortDuration = Duration(milliseconds: 200);
  static const Duration mediumDuration = Duration(milliseconds: 300);
  static const Duration longDuration = Duration(milliseconds: 500);
  static const Duration veryLongDuration = Duration(milliseconds: 800);

  // Curve constants
  static const Curve defaultCurve = Curves.easeInOut;
  static const Curve bounceCurve = Curves.elasticOut;
  static const Curve smoothCurve = Curves.easeOutCubic;
  static const Curve sharpCurve = Curves.easeInOutCubic;

  /// Standard animation controller duration
  static const Duration standardDuration = mediumDuration;

  /// Breathing animation duration (for calm/relaxed emotions)
  static const Duration breathingDuration = Duration(seconds: 3);

  /// Pulsing animation duration (for stress/anxiety)
  static const Duration pulsingDuration = Duration(milliseconds: 1000);

  /// Face scan animation duration
  static const Duration faceScanDuration = Duration(seconds: 3);

  /// Emotion reveal animation duration
  static const Duration emotionRevealDuration = Duration(milliseconds: 600);

  /// Get animation curve based on emotion type
  static Curve getEmotionCurve(String emotionType) {
    switch (emotionType.toLowerCase()) {
      case 'calm':
      case 'relaxed':
        return Curves.easeInOut;
      case 'stress':
      case 'anxiety':
        return Curves.easeInOut;
      case 'happiness':
      case 'motivation':
        return Curves.elasticOut;
      default:
        return defaultCurve;
    }
  }

  /// Get animation duration based on emotion type
  static Duration getEmotionDuration(String emotionType) {
    switch (emotionType.toLowerCase()) {
      case 'calm':
      case 'relaxed':
        return breathingDuration;
      case 'stress':
      case 'anxiety':
        return pulsingDuration;
      default:
        return mediumDuration;
    }
  }
}