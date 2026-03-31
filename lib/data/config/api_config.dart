import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

/// API Configuration for Emotion Detection Backend
class ApiConfig {
  /// Production mode flag
  /// Set [forceProductionMode] to true to test AWS backend in debug mode
  static const bool forceProductionMode = true;

  static bool get useProductionMode => kReleaseMode || forceProductionMode;

  /// AWS/Production backend URL
  static String get productionBaseUrl =>
      _getEnv('API_BASE_URL') ??
      'http://emotion-api-env.eba-xkn3z6bd.us-east-1.elasticbeanstalk.com';

  /// Safe helper to get environment variables before/during initialization
  static String? _getEnv(String key) {
    try {
      if (dotenv.isInitialized) {
        return dotenv.env[key];
      }
    } catch (_) {
      // Return null if not initialized or fails
    }
    return null;
  }

  /// Local development URL
  static String _manualBaseUrlInternal = '';

  static String get manualBaseUrl {
    if (forceProductionMode) return '';
    return _manualBaseUrlInternal;
  }

  static set manualBaseUrl(String value) {
    if (forceProductionMode) return;
    _manualBaseUrlInternal = value;
  }

  // Base URL for the API server
  // ALWAYS uses AWS when forceProductionMode is true
  // Base URL for the API server
  // ALWAYS uses AWS when forceProductionMode is true
  static String get baseUrl {
    if (forceProductionMode || kReleaseMode) {
      final url = productionBaseUrl;
      if (!kReleaseMode) {
        debugPrint('[API_CONFIG] 🔒 FORCED AWS PRODUCTION URL: $url');
      }
      return url;
    }

    // --- Development Mode Logic (Only reached if forceProductionMode is FALSE) ---

    // Use Manual Local URL if set
    if (manualBaseUrl.isNotEmpty) {
      debugPrint('[API_CONFIG] 🔧 USING MANUAL (LOCAL) URL: $manualBaseUrl');
      return manualBaseUrl;
    }

    // Fallback for Emulator
    if (Platform.isAndroid) return 'http://10.0.2.2:5000';
    return 'http://localhost:5000';
  }

  // API endpoints
  static const String detectEmotionEndpoint = '/api/v1/detect-emotion';
  static const String healthEndpoint = '/api/v1/health';
  static const String infoEndpoint = '/api/v1/info';
  static const String statsEndpoint = '/api/v1/stats';

  // API Key Configuration
  /// API key for authentication
  ///
  /// **Default Test Key**: `emotion-api-flutter-mobile-abcdef1234567890`
  ///
  /// **For Production**: Set this from environment variable or secure storage
  ///
  /// **Available Test Keys**:
  /// - `emotion-api-test-1234567890abcdef` - Default test key
  /// - `emotion-api-postman-9876543210fedcba` - Postman testing
  /// - `emotion-api-flutter-mobile-abcdef1234567890` - Flutter mobile app
  static const String apiKey = 'emotion-api-flutter-mobile-abcdef1234567890';

  // Gemini Configuration
  static String get geminiKey => _getEnv('GEMINI_API_KEY') ?? '';
  static String get geminiModel => _getEnv('GEMINI_API_KEY') != null ? (_getEnv('GEMINI_MODEL') ?? 'gemini-flash-latest') : 'gemini-flash-latest';

  // Request timeout (increased for Render cold starts and slow connections)
  static Duration get timeout {
    // Render cold starts can take up to 30 seconds, so increase timeout in production
    if (useProductionMode) {
      return const Duration(seconds: 35); // Allow time for Render cold start
    }
    return const Duration(seconds: 8); // Local development timeout
  }

  static const Duration healthCheckTimeout = Duration(
    seconds: 15,
  ); // Increased for slow connections and EB spin-up

  // Retry configuration
  static const int maxRetries = 3;
  static const Duration retryInitialDelay = Duration(milliseconds: 500);
  static const double retryBackoffMultiplier = 2.0;

  // Connection retry configuration
  static int get maxConnectionRetries {
    // More retries for Render due to cold starts
    if (useProductionMode) {
      return 3; // Additional retry for Render cold starts
    }
    return 2; // Local development retries
  }

  static const Duration connectionRetryDelay = Duration(
    seconds: 2,
  ); // Delay between connection retries

  // Sampling configuration
  static const Duration samplingInterval = Duration(milliseconds: 800);

  // Feature flags
  static const bool useMockFallback =
      false; // Fallback to mock service if API fails
  static const bool enableApiService = true; // Toggle API service on/off

  // Performance settings
  static const int imageQuality = 80; // JPEG quality (0-100)
  static const int maxImageWidth = 320; // Max width for image compression
  static const int maxImageHeight = 240; // Max height for image compression

  // Confidence thresholds
  // Lowered from 0.3 to 0.25 to better detect all emotions (angry, sad, fear, disgust, surprise)
  // Some emotions naturally have lower confidence scores from the model
  static const double minConfidenceThreshold =
      0.25; // Minimum confidence to accept detection (lowered for better emotion detection)
  static const double highConfidenceThreshold =
      0.7; // High confidence threshold

  // Emotion-specific confidence thresholds (for emotions that are harder to detect)
  // These emotions may have lower natural confidence scores from the model
  static const Map<String, double> emotionSpecificThresholds = {
    'angry': 0.20, // Angry expressions can be subtle
    'sad': 0.22, // Sad expressions can be subtle
    'fear': 0.22, // Fear expressions can be subtle
    'disgust': 0.20, // Disgust expressions can be subtle
    'surprise': 0.25, // Surprise can be brief
    'happy': 0.25, // Happy is usually well-detected
    'neutral':
        0.30, // Neutral should have higher threshold to avoid false positives
  };

  /// Get confidence threshold for a specific emotion
  /// Returns emotion-specific threshold if available, otherwise uses minConfidenceThreshold
  static double getConfidenceThresholdForEmotion(String emotionType) {
    final lowerEmotion = emotionType.toLowerCase();
    return emotionSpecificThresholds[lowerEmotion] ?? minConfidenceThreshold;
  }

  // Get full URL for endpoint
  static String getEndpointUrl(String endpoint) {
    return '$baseUrl$endpoint';
  }

  /// Get connection info for debugging
  static String getConnectionInfo() {
    final isPhysicalDevice = _isPhysicalDevice();
    return 'API URL: $baseUrl\n'
        'Platform: ${Platform.operatingSystem}\n'
        'Device Type: ${isPhysicalDevice ? "Physical Device" : "Emulator/Simulator"}\n'
        'Manual Override: ${manualBaseUrl.isNotEmpty ? manualBaseUrl : "None (using auto-detection)"}\n'
        '${isPhysicalDevice && manualBaseUrl.isEmpty ? "⚠️ WARNING: Physical device detected but manualBaseUrl is not set!\n   Set manualBaseUrl to your computer's IP address (e.g., http://192.168.1.100:5000)" : ""}';
  }

  /// Detect if running on physical device vs emulator/simulator
  /// This is a heuristic - physical devices typically have different characteristics
  static bool _isPhysicalDevice() {
    if (Platform.isAndroid) {
      // Android: Check if we're using manual override (indicates physical device setup)
      // Or check environment variables that differ between emulator and device
      // For now, we'll assume manual override means physical device
      // Note: manualBaseUrl will be empty if forceProductionMode is true
      return manualBaseUrl.isNotEmpty;
    }
    // iOS: Similar logic
    return false;
  }

  /// Get setup instructions based on device type
  static String getSetupInstructions() {
    if (manualBaseUrl.isNotEmpty) {
      return 'Using manual URL: $manualBaseUrl\n'
          'Make sure:\n'
          '1. API server is running: cd "Emotion Detection" && python api_server.py\n'
          '2. Your device and computer are on the same Wi-Fi network\n'
          '3. Firewall allows connections on port 5000';
    }

    if (Platform.isAndroid) {
      return 'Using Android emulator URL: http://10.0.2.2:5000\n'
          'For physical devices, set manualBaseUrl in api_config.dart\n'
          'Make sure API server is running: cd "Emotion Detection" && python api_server.py';
    }

    return 'Make sure API server is running: cd "Emotion Detection" && python api_server.py';
  }

  // Validate URL format
  static bool isValidUrl(String url) {
    try {
      final uri = Uri.parse(url);
      return uri.hasScheme &&
          (uri.scheme == 'http' || uri.scheme == 'https') &&
          uri.hasAuthority;
    } catch (e) {
      return false;
    }
  }
}
