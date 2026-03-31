import 'dart:async';
import 'package:flutter/foundation.dart';
import 'dart:convert';
import 'dart:io';
import 'dart:ui';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;
import '../config/api_config.dart';
import '../models/emotion_model.dart';
import '../utils/mock_data.dart';

class EmotionApiService {
  static final EmotionApiService _instance = EmotionApiService._internal();
  factory EmotionApiService() => _instance;
  EmotionApiService._internal();

  bool _isApiAvailable = true;
  DateTime? _lastApiCheck;
  static const Duration _apiCheckInterval = Duration(seconds: 30);

  String _connectionStatus = 'unknown';
  String? _lastConnectionError;

  final _statusController = StreamController<String>.broadcast();
  Stream<String> get statusStream => _statusController.stream;

  _PendingRequest? _latestPendingRequest;
  bool _isProcessing = false;

  String get connectionStatus => _connectionStatus;
  String? get lastConnectionError => _lastConnectionError;

  EmotionModel? _cachedResponse;
  DateTime? _cacheTimestamp;
  static const Duration _cacheValidity = Duration(milliseconds: 150);

  Future<EmotionModel?> detectEmotionFromCamera({
    required Uint8List imageBytes,
    String? previousEmotionType,
    DateTime? previousEmotionTime,
    bool skipCompression = false,
  }) async {
    if (_cachedResponse != null && _cacheTimestamp != null) {
      final cacheAge = DateTime.now().difference(_cacheTimestamp!);
      if (cacheAge < _cacheValidity) {
        return _cachedResponse;
      }
    }

    if (!ApiConfig.enableApiService) {
      _connectionStatus = 'disconnected';
      _lastConnectionError = 'API service is disabled in configuration';
      return await _fallbackToMock(previousEmotionType);
    }

    if (!_isApiAvailable && _shouldRetryApiCheck()) {
      final isAvailable = await _checkApiHealth();
      if (!isAvailable && ApiConfig.useMockFallback) {
        // Keep the specific error status (timeout/error) set by _checkApiHealth
        final isRender =
            ApiConfig.useProductionMode &&
            ApiConfig.baseUrl.contains('onrender.com');
        _lastConnectionError = isRender
            ? 'Render service may be starting (cold start can take 30-60s). Using mock service temporarily.'
            : 'API server not accessible. Using mock service.';
        return await _fallbackToMock(previousEmotionType);
      }
      _isApiAvailable = isAvailable;
      // status is already updated inside _checkApiHealth
    }

    final completer = Completer<EmotionModel?>();
    final request = _PendingRequest(
      imageBytes: imageBytes,
      previousEmotionType: previousEmotionType,
      previousEmotionTime: previousEmotionTime,
      completer: completer,
      skipCompression: skipCompression,
    );

    debugPrint(
      '[API] Processing new request. Image size: ${(imageBytes.length / 1024).toStringAsFixed(1)} KB',
    );
    _latestPendingRequest = request;
    _processLatestRequest();

    return completer.future;
  }

  Future<void> _processLatestRequest() async {
    if (_isProcessing) return;

    while (_latestPendingRequest != null) {
      _isProcessing = true;
      final request = _latestPendingRequest!;
      _latestPendingRequest = null;

      try {
        final result = await _detectEmotionWithRetry(
          request.imageBytes,
          previousEmotionType: request.previousEmotionType,
          previousEmotionTime: request.previousEmotionTime,
          skipCompression: request.skipCompression,
        );
        if (!request.completer.isCompleted) {
          request.completer.complete(result);
        }
      } catch (e) {
        if (!request.completer.isCompleted) {
          request.completer.complete(null);
        }
      }
    }
    _isProcessing = false;
  }

  /// Detect emotion with retry logic
  Future<EmotionModel?> _detectEmotionWithRetry(
    Uint8List imageBytes, {
    String? previousEmotionType,
    DateTime? previousEmotionTime,
    bool skipCompression = false,
  }) async {
    Duration delay = ApiConfig.retryInitialDelay;

    for (int attempt = 0; attempt < ApiConfig.maxRetries; attempt++) {
      try {
        final result = await _detectEmotionApiCall(
          imageBytes,
          previousEmotionType: previousEmotionType,
          previousEmotionTime: previousEmotionTime,
          skipCompression: skipCompression,
        );

        if (result != null) return result;
        return null;
      } on TimeoutException {
        final isRender =
            ApiConfig.useProductionMode &&
            ApiConfig.baseUrl.contains('onrender.com');
        _lastConnectionError = isRender
            ? 'Render service may be starting (cold start). Retrying...'
            : 'Connection timeout.';
        _connectionStatus = 'timeout';

        if (attempt < ApiConfig.maxRetries - 1) {
          await Future.delayed(delay);
          delay = Duration(
            milliseconds:
                (delay.inMilliseconds * ApiConfig.retryBackoffMultiplier)
                    .round(),
          );
        }
      } on SocketException {
        _lastConnectionError = 'Network error: Cannot connect to API.';
        _connectionStatus = 'error';
        if (attempt < ApiConfig.maxRetries - 1) {
          await Future.delayed(delay);
          delay = Duration(
            milliseconds:
                (delay.inMilliseconds * ApiConfig.retryBackoffMultiplier)
                    .round(),
          );
        } else {
          rethrow;
        }
      } catch (e) {
        _lastConnectionError = 'Unexpected error: $e';
        _connectionStatus = 'error';
        debugPrint('[API] DETECT EMOTION ERROR: $_lastConnectionError');
        return null;
      }
    }
    return null;
  }

  Future<EmotionModel?> _detectEmotionApiCall(
    Uint8List imageBytes, {
    String? previousEmotionType,
    DateTime? previousEmotionTime,
    bool skipCompression = false,
  }) async {
    final Uint8List compressedImage = skipCompression
        ? imageBytes
        : await _compressImage(imageBytes);
    final base64Image = base64Encode(compressedImage);

    final url = Uri.parse(
      ApiConfig.getEndpointUrl(ApiConfig.detectEmotionEndpoint),
    );
    final requestBody = jsonEncode({
      'image': base64Image,
      'format': 'jpeg',
      'timestamp': DateTime.now().toIso8601String(),
    });

    final startTime = DateTime.now();
    final response = await http
        .post(
          url,
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': ApiConfig.apiKey,
          },
          body: requestBody,
        )
        .timeout(ApiConfig.timeout);

    final duration = DateTime.now().difference(startTime);
    debugPrint(
      '[API] Response received in ${duration.inMilliseconds}ms. Status: ${response.statusCode}',
    );

    if (response.statusCode == 200) {
      final responseData = jsonDecode(response.body) as Map<String, dynamic>;
      if (responseData['success'] == true) {
        return _parseApiResponse(
          responseData,
          previousEmotionType: previousEmotionType,
          previousEmotionTime: previousEmotionTime,
        );
      } else {
        debugPrint(
          '[API] Request failed. Success=false. Response: ${response.body}',
        );
      }
    } else {
      debugPrint('[API] HTTP Error ${response.statusCode}: ${response.body}');
    }
    return null;
  }

  /// Parse API response to EmotionModel
  /// Uses backend emotions directly (no mapping needed)
  /// Validates response format matches backend exactly
  EmotionModel? _parseApiResponse(
    Map<String, dynamic> responseData, {
    String? previousEmotionType,
    DateTime? previousEmotionTime,
  }) {
    try {
      // Validate response structure
      if (responseData['success'] != true || responseData['emotion'] == null) {
        debugPrint(
          '[API] Invalid response structure: success=${responseData['success']}, emotion=${responseData['emotion']}',
        );
        return null;
      }

      final emotionData = responseData['emotion'] as Map<String, dynamic>;

      // Extract and validate emotion type
      if (emotionData['type'] == null) {
        debugPrint('[API] Missing emotion type in response');
        return null;
      }
      final backendEmotion = (emotionData['type'] as String).toLowerCase();

      // Validate backend emotion is one of the 7 trained emotions
      const validEmotions = [
        'angry',
        'disgust',
        'fear',
        'happy',
        'neutral',
        'sad',
        'surprise',
      ];
      if (!validEmotions.contains(backendEmotion)) {
        debugPrint(
          '[API] Invalid backend emotion: $backendEmotion (not in trained set)',
        );
        return null;
      }

      // Extract and validate confidence
      if (emotionData['confidence'] == null) {
        debugPrint('[API] Missing confidence in response');
        return null;
      }
      final confidence = (emotionData['confidence'] as num).toDouble();

      // Validate confidence range (0.0-1.0)
      if (confidence < 0.0 || confidence > 1.0) {
        debugPrint(
          '[API] Invalid confidence range: $confidence (expected 0.0-1.0)',
        );
        return null;
      }

      // Check confidence threshold
      if (confidence < ApiConfig.minConfidenceThreshold) {
        debugPrint(
          '[API] Low confidence: $confidence < ${ApiConfig.minConfidenceThreshold} (threshold)',
        );
        return null; // Low confidence, skip
      }

      // Use backend's intensity if available, otherwise calculate from confidence
      int intensity;
      if (emotionData['intensity'] != null) {
        final backendIntensity = (emotionData['intensity'] as num).toInt();
        // Validate intensity range (0-100)
        intensity = backendIntensity.clamp(30, 100);
      } else {
        // Fallback: calculate from confidence (0.0-1.0 -> 30-100)
        final baseIntensity = (confidence * 100).round();
        intensity = baseIntensity.clamp(30, 100);
      }

      // Get suggestions (reuse from mock service)
      final suggestions = _getSuggestionsForEmotion(backendEmotion);

      // Parse face detection data
      final faceDetected = emotionData['face_detected'] as bool? ?? false;
      final numFaces = emotionData['num_faces'] as int? ?? 0;

      // Parse face bounds from API response
      // API returns: {x, y, width, height} in image pixel coordinates
      Rect? faceBounds;
      if (faceDetected && emotionData['face_bounds'] != null) {
        try {
          final bounds = emotionData['face_bounds'] as Map<String, dynamic>;
          final x = (bounds['x'] as num?)?.toDouble() ?? 0.0;
          final y = (bounds['y'] as num?)?.toDouble() ?? 0.0;
          final width = (bounds['width'] as num?)?.toDouble() ?? 0.0;
          final height = (bounds['height'] as num?)?.toDouble() ?? 0.0;

          if (width > 0 && height > 0) {
            // Create Rect from API coordinates (image pixel coordinates)
            // Note: These are in image coordinate space, will be converted to screen coordinates in UI
            faceBounds = Rect.fromLTWH(x, y, width, height);
            debugPrint(
              '[API] Face bounds: x=$x, y=$y, width=$width, height=$height',
            );
          }
        } catch (e) {
          debugPrint('[API] Error parsing face bounds: $e');
        }
      }

      debugPrint(
        '[API] Parsed emotion: $backendEmotion (confidence: ${(confidence * 100).toStringAsFixed(1)}%, intensity: $intensity%, faceDetected: $faceDetected, numFaces: $numFaces)',
      );

      return EmotionModel(
        id: DateTime.now().toIso8601String(),
        type: backendEmotion, // Use backend emotion directly
        intensity: intensity,
        timestamp: DateTime.now(),
        detectionMethod: 'camera',
        suggestions: suggestions,
        journallingPrompt: MockEmotionService.getJournallingPromptForEmotion(
          backendEmotion,
        ),
        faceDetected: faceDetected,
        faceBounds: faceBounds,
        numFaces: numFaces,
      );
    } catch (e) {
      debugPrint('[API] Error parsing API response: $e');
      debugPrint('[API] Response data: $responseData');
      return null;
    }
  }

  Future<Uint8List> _compressImage(Uint8List imageBytes) async {
    final startTime = DateTime.now();
    final result = await compute(
      _compressImageIsolate,
      _CompressionData(
        imageBytes: imageBytes,
        maxWidth: ApiConfig.maxImageWidth,
        maxHeight: ApiConfig.maxImageHeight,
        quality: ApiConfig.imageQuality,
      ),
    );
    final duration = DateTime.now().difference(startTime);
    debugPrint(
      '[ImageUtils] Compression took ${duration.inMilliseconds}ms. Original: ${(imageBytes.length / 1024).toStringAsFixed(1)} KB, Compressed: ${(result.length / 1024).toStringAsFixed(1)} KB',
    );
    return result;
  }

  /// Get suggestions for emotion (reuse from mock service)
  List<String> _getSuggestionsForEmotion(String emotionType) {
    return MockEmotionService.getSuggestionsForEmotion(emotionType);
  }

  /// Pre-flight check: Verify API server is accessible before starting detection
  /// Returns true if API is accessible, false otherwise
  /// Updates connection status accordingly
  Future<bool> preflightCheck() async {
    debugPrint('[API] Performing pre-flight health check...');
    _connectionStatus = 'checking';
    _lastConnectionError = null;

    final isHealthy = await _checkApiHealth();

    if (isHealthy) {
      _connectionStatus = 'connected';
      _isApiAvailable = true;
      debugPrint('[API] ✅ Pre-flight check passed - API server is accessible');
    } else {
      _connectionStatus = 'disconnected';
      _isApiAvailable = false;
      debugPrint(
        '[API] ❌ Pre-flight check failed - API server is not accessible',
      );
      debugPrint('[API] ${ApiConfig.getSetupInstructions()}');
    }

    return isHealthy;
  }

  /// Check API health (public method for external use)
  /// Check API health (public method for external use)
  Future<bool> checkApiHealth() async {
    final isAvailable = await _checkApiHealth();
    _isApiAvailable = isAvailable;
    // Status is set inside _checkApiHealth

    _notifyStatusChange();
    return isAvailable;
  }

  /// Notify status change (for syncing with controllers)
  void _notifyStatusChange() {
    if (!_statusController.isClosed) {
      _statusController.add(_connectionStatus);
    }
  }

  /// Check API health (internal)
  Future<bool> _checkApiHealth() async {
    final url = Uri.parse(ApiConfig.getEndpointUrl(ApiConfig.healthEndpoint));
    debugPrint('[API] 🔍 Checking health at $url');
    try {
      _connectionStatus = 'checking';
      _lastConnectionError = null;

      // Check if using Render (might need longer timeout for cold starts)
      final isRender =
          ApiConfig.useProductionMode &&
          ApiConfig.baseUrl.contains('onrender.com');
      final timeout = isRender
          ? const Duration(seconds: 10) // Longer timeout for Render cold starts
          : ApiConfig.healthCheckTimeout;

      final response = await http.get(url).timeout(timeout);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final isHealthy =
            data['status'] == 'healthy' &&
            data['model_loaded'] == true &&
            data['cascade_loaded'] == true;

        if (isHealthy) {
          _connectionStatus = 'connected';
          _lastConnectionError = null;
          debugPrint('[API] Health check result: ✅ Connected and healthy');
        } else {
          _connectionStatus = 'error';
          _lastConnectionError =
              'API health check failed: Model or cascade not loaded';
          debugPrint(
            '[API] Health check result: ❌ Unhealthy (model or cascade not loaded). Response body: ${response.body}',
          );
        }
        _notifyStatusChange();
        return isHealthy;
      }

      _connectionStatus = 'error';
      _lastConnectionError =
          'API health check failed: HTTP ${response.statusCode} - ${response.reasonPhrase}';
      debugPrint(
        '[API] Health check failed: status ${response.statusCode} (${response.reasonPhrase}). Body: ${response.body}',
      );
      _notifyStatusChange();
      return false;
    } on TimeoutException catch (e) {
      _connectionStatus = 'timeout';
      _lastConnectionError =
          'Health check timeout. API server may not be running at ${ApiConfig.baseUrl}';
      debugPrint('[API] Health check timeout: $e');
      debugPrint(
        '[API] 💡 Tip: Make sure the API server is running. Start it with: cd "Emotion Detection" && python api_server.py',
      );
      _notifyStatusChange();
      return false;
    } on SocketException catch (e) {
      _connectionStatus = 'error';
      _lastConnectionError =
          'Cannot connect to API server at ${ApiConfig.baseUrl}. Check if server is running.';
      debugPrint('[API] Health check network error: $e');
      debugPrint(
        '[API] 💡 Tip: For physical devices, set ApiConfig.manualBaseUrl to your computer\'s IP address',
      );
      _notifyStatusChange();
      return false;
    } catch (e) {
      _connectionStatus = 'error';
      _lastConnectionError = 'Health check error: ${e.toString()}';
      debugPrint('[API] Health check error: $e');
      _notifyStatusChange();
      return false;
    }
  }

  /// Check if we should retry API check
  bool _shouldRetryApiCheck() {
    if (_lastApiCheck == null) return true;
    return DateTime.now().difference(_lastApiCheck!) > _apiCheckInterval;
  }

  /// Fallback to mock service
  Future<EmotionModel> _fallbackToMock(String? previousEmotionType) async {
    return await MockEmotionService.detectEmotionFromCamera(
      previousEmotionType: previousEmotionType,
    );
  }
}

/// Pending request data structure
class _PendingRequest {
  final Uint8List imageBytes;
  final String? previousEmotionType;
  final DateTime? previousEmotionTime;
  final Completer<EmotionModel?> completer;
  final bool skipCompression;

  _PendingRequest({
    required this.imageBytes,
    this.previousEmotionType,
    this.previousEmotionTime,
    required this.completer,
    this.skipCompression = false,
  });
}

/// Data class for compression isolate
class _CompressionData {
  final Uint8List imageBytes;
  final int maxWidth;
  final int maxHeight;
  final int quality;

  _CompressionData({
    required this.imageBytes,
    required this.maxWidth,
    required this.maxHeight,
    required this.quality,
  });
}

/// Isolate function for image compression
Uint8List _compressImageIsolate(_CompressionData data) {
  try {
    // Decode image
    final image = img.decodeImage(data.imageBytes);
    if (image == null) {
      return data.imageBytes;
    }

    // Resize if too large
    img.Image resizedImage = image;
    if (image.width > data.maxWidth || image.height > data.maxHeight) {
      final aspectRatio = image.width / image.height;
      int newWidth = data.maxWidth;
      int newHeight = (newWidth / aspectRatio).round();

      if (newHeight > data.maxHeight) {
        newHeight = data.maxHeight;
        newWidth = (newHeight * aspectRatio).round();
      }

      resizedImage = img.copyResize(
        image,
        width: newWidth,
        height: newHeight,
        interpolation: img.Interpolation.linear,
      );
    }

    // Encode as JPEG
    final compressedBytes = Uint8List.fromList(
      img.encodeJpg(resizedImage, quality: data.quality),
    );

    return compressedBytes;
  } catch (e) {
    debugPrint('Error in compression isolate: $e');
    return data.imageBytes;
  }
}
