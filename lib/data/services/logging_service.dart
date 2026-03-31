import 'dart:developer' as developer;

/// Centralized logging service for production
/// Provides structured logging with levels and optional remote logging
class LoggingService {
  static final LoggingService _instance = LoggingService._internal();
  factory LoggingService() => _instance;
  LoggingService._internal();

  // Log levels
  static const String levelDebug = 'DEBUG';
  static const String levelInfo = 'INFO';
  static const String levelWarning = 'WARNING';
  static const String levelError = 'ERROR';

  // Enable/disable logging
  static bool _enabled = true;
  static bool _debugEnabled = true; // Disable in production

  /// Enable or disable logging
  static void setEnabled(bool enabled) {
    _enabled = enabled;
  }

  /// Enable or disable debug logging
  static void setDebugEnabled(bool enabled) {
    _debugEnabled = enabled;
  }

  /// Log debug message
  static void debug(String message, {String? tag, Map<String, dynamic>? data}) {
    if (!_enabled || !_debugEnabled) return;
    _log(levelDebug, message, tag: tag, data: data);
  }

  /// Log info message
  static void info(String message, {String? tag, Map<String, dynamic>? data}) {
    if (!_enabled) return;
    _log(levelInfo, message, tag: tag, data: data);
  }

  /// Log warning message
  static void warning(
    String message, {
    String? tag,
    Map<String, dynamic>? data,
    Object? error,
  }) {
    if (!_enabled) return;
    _log(levelWarning, message, tag: tag, data: data, error: error);
  }

  /// Log error message
  static void error(
    String message, {
    String? tag,
    Map<String, dynamic>? data,
    Object? error,
    StackTrace? stackTrace,
  }) {
    if (!_enabled) return;
    _log(
      levelError,
      message,
      tag: tag,
      data: data,
      error: error,
      stackTrace: stackTrace,
    );
  }

  /// Internal logging method
  static void _log(
    String level,
    String message, {
    String? tag,
    Map<String, dynamic>? data,
    Object? error,
    StackTrace? stackTrace,
  }) {
    final logTag = tag ?? 'App';

    // Format log message
    final logMessage = '[$level] [$logTag] $message';

    // Add data if provided
    if (data != null && data.isNotEmpty) {
      final dataStr = data.entries.map((e) => '${e.key}=${e.value}').join(', ');
      developer.log('$logMessage | Data: $dataStr', name: logTag);
    } else {
      developer.log(logMessage, name: logTag);
    }

    // Log error if provided
    if (error != null) {
      developer.log(
        'Error: $error',
        name: logTag,
        error: error,
        stackTrace: stackTrace,
      );
    }

    // In production, you could send logs to remote service here
    // Example: Firebase Crashlytics, Sentry, etc.
    // _sendToRemoteService(level, message, tag: tag, data: data, error: error);
  }
}
