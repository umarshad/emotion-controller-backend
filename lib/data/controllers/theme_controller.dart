import 'package:flutter/material.dart';
import 'package:get/get.dart';

/// Controller for managing app theme (light/dark mode)
class ThemeController extends GetxController {
  // Current theme mode - forced to light mode only
  final Rx<ThemeMode> themeMode = ThemeMode.light.obs;

  @override
  void onInit() {
    super.onInit();
    // Force light mode only - dark mode removed
    themeMode.value = ThemeMode.light;
  }

  /// Check if dark mode is currently active
  /// Always returns false since dark mode is disabled
  bool get isDarkMode {
    return false; // Dark mode is disabled
  }

  /// Toggle between light and dark mode
  /// Disabled - app only uses light mode
  void toggleTheme() {
    // No-op: Dark mode is disabled, always use light mode
    themeMode.value = ThemeMode.light;
  }

  /// Set theme mode
  /// Always forces light mode regardless of input
  void setThemeMode(ThemeMode mode) {
    themeMode.value = ThemeMode.light; // Always force light mode
  }

  /// Get current theme mode as string
  /// Always returns 'Light' since dark mode is disabled
  String get currentThemeMode {
    return 'Light'; // Always light mode
  }
}


