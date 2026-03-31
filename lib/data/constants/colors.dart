import 'package:flutter/material.dart';

/// Color scheme for Emotion Controller
/// Primary: Teal
/// Secondary: Lavender
/// Background: Warm Off-White (Light) / Dark Gray (Dark)
/// Accent: Low-saturation Coral OR Mint
class AppColors {
  // Primary Colors - Teal
  static const Color primaryBlue = Color(0xFF6B9BD1);
  static const Color primaryTeal = Color(0xFF4ECDC4);
  static const Color primary = Color(0xFF4ECDC4); // Pure Teal

  // Secondary Colors - Lavender
  static const Color secondaryGreen = Color(0xFFA8D5BA);
  static const Color secondaryLavender = Color(0xFFB8A9D5);
  static const Color secondary = Color(0xFFB8A9D5); // Pure Lavender

  // Background - Warm Off-White (NOT pure white) - Light mode
  // These are used in ThemeData definitions
  static const Color lightBackground = Color(0xFFF8F6F2);
  static const Color lightSurface = Color(0xFFF5F3EF);
  static const Color lightCardBackground = Color(0xFFFAF9F5);

  // Accent Colors - Low-saturation Coral / Mint
  static const Color accentCoral = Color(0xFFF5A097);
  static const Color accentMint = Color(0xFFB5E5D8);
  static const Color accent = Color(0xFFE5B5C8); // Soft accent blend

  // Text Colors - Light mode
  static const Color lightTextPrimary = Color(0xFF2C3E50);
  static const Color lightTextSecondary = Color(0xFF7F8C8D);
  static const Color lightTextLight = Color(0xFFBDC3C7);

  // Emotion-specific colors (7 backend emotions)
  static const Color emotionAngry = Color(0xFFE74C3C);
  static const Color emotionDisgust = Color(0xFF8B4513);
  static const Color emotionFear = Color(0xFF8E44AD);
  static const Color emotionHappy = Color(0xFFF39C12);
  static const Color emotionNeutral = Color(0xFF95A5A6);
  static const Color emotionSad = Color(0xFF5D6D7E);
  static const Color emotionSurprise = Color(0xFF3498DB);

  // UI Colors
  static const Color success = Color(0xFF27AE60);
  static const Color warning = Color(0xFFF39C12);
  static const Color error = Color(0xFFE74C3C);
  static const Color info = Color(0xFF3498DB);

  // Border and Divider - Light mode
  static const Color lightBorder = Color(0xFFE8E8E8);
  static const Color lightDivider = Color(0xFFE0E0E0);

  // Shadow - Light mode
  static const Color lightShadow = Color(0x1A000000);

  // ========== DARK MODE COLORS ==========

  // Dark Background
  static const Color darkBackground = Color(0xFF121212);
  static const Color darkSurface = Color(0xFF1E1E1E);
  static const Color darkCardBackground = Color(0xFF2C2C2C);

  // Dark Text Colors
  static const Color darkTextPrimary = Color(0xFFE0E0E0);
  static const Color darkTextSecondary = Color(0xFFB0B0B0);
  static const Color darkTextLight = Color(0xFF808080);

  // Dark Border and Divider
  static const Color darkBorder = Color(0xFF3A3A3A);
  static const Color darkDivider = Color(0xFF404040);

  // Dark Shadow
  static const Color darkShadow = Color(0x40000000);

  // ========== THEME-AWARE GETTERS ==========
  // Simplified to always return light mode colors (dark mode disabled)

  /// Get background color - always returns light mode
  static Color get background => lightBackground;

  /// Get surface color - always returns light mode
  static Color get surface => lightSurface;

  /// Get card background color - always returns light mode
  static Color get cardBackground => lightCardBackground;

  /// Get primary text color - always returns light mode
  static Color get textPrimary => lightTextPrimary;

  /// Get secondary text color - always returns light mode
  static Color get textSecondary => lightTextSecondary;

  /// Get light text color - always returns light mode
  static Color get textLight => lightTextLight;

  /// Get border color - always returns light mode
  static Color get border => lightBorder;

  /// Get divider color - always returns light mode
  static Color get divider => lightDivider;

  /// Get shadow color - always returns light mode
  static Color get shadow => lightShadow;
}

/// Get emotion color by emotion type
/// Returns color for the 7 backend-trained emotions
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
