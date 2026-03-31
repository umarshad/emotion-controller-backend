// ignore_for_file: avoid_print
import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart' as cam;
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart'; // For DeviceOrientation
import 'package:flutter/widgets.dart'; // For app lifecycle
import 'package:get/get.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart'
    as img; // For cropping/format conversion if needed

import '../config/api_config.dart';
import '../models/emotion_model.dart';
import '../services/emotion_api_service.dart';
import '../../view/camera/image_preview_screen.dart';
import 'emotion_controller.dart';

enum CameraState {
  initializing,
  permissionDenied,
  ready,
  error,
  background, // App is in background
}

class CameraController extends GetxController with WidgetsBindingObserver {
  // --- State ---
  final Rx<CameraState> cameraState = CameraState.initializing.obs;
  final RxString errorMessage = ''.obs;

  bool _isApiBusy = false;
  DateTime? _lastFaceDetectedTime;

  cam.CameraController? _cameraController;
  cam.CameraController? get cameraController => _cameraController;
  List<cam.CameraDescription> _cameras = [];
  final RxInt currentCameraIndex = 0.obs;
  final RxInt currentCameraLens = 1.obs; // 0 = Back, 1 = Front

  late final FaceDetector _faceDetector;

  final Rx<Rect?> detectedFaceBounds = Rx<Rect?>(null);
  final RxBool isFaceDetected = false.obs;

  final Rx<EmotionModel?> currentEmotion = Rx<EmotionModel?>(null);
  final RxBool isAnalyzingEmotion = false.obs;

  final RxString apiStatus = 'Checking...'.obs;
  final Rx<Color> emotionColor = const Color(0xFF4CAF50).obs;

  // --- Internals ---
  bool _isProcessingFrame = false;
  DateTime? _lastFaceDetectionTime;
  static const Duration _faceDetectionInterval = Duration(
    milliseconds: 300,
  ); // 3-4 FPS for face detection throttling

  bool _isStopping = false;

  DateTime? _lastApiCallTime;
  StreamSubscription? _apiStatusSubscription;

  @override
  void onInit() {
    super.onInit();
    WidgetsBinding.instance.addObserver(this);

    _faceDetector = FaceDetector(
      options: FaceDetectorOptions(
        enableContours: false,
        enableClassification: false,
        enableLandmarks: false,
        enableTracking: true,
        minFaceSize: 0.15,
        performanceMode: FaceDetectorMode.fast,
      ),
    );

    _apiStatusSubscription = EmotionApiService().statusStream.listen((status) {
      if (status == 'connected') {
        apiStatus.value = 'Connected';
      } else if (status == 'disconnected') {
        apiStatus.value = 'Offline';
      } else if (status == 'checking') {
        apiStatus.value = 'Checking...';
      } else if (status == 'timeout') {
        apiStatus.value = 'Timeout';
      } else {
        apiStatus.value = 'Error';
      }
    });

    // Trigger initial check
    EmotionApiService().checkApiHealth();

    _initializeCamera();
  }

  @override
  void onClose() {
    WidgetsBinding.instance.removeObserver(this);
    _stopStream();
    _apiStatusSubscription?.cancel();
    _cameraController?.dispose();
    _faceDetector.close();
    super.onClose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Handle backgrounding to release camera
    if (cameraState.value == CameraState.permissionDenied ||
        cameraState.value == CameraState.error) {
      return;
    }

    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      // Background
      debugPrint('[CameraController] App backgrounded. Releasing camera.');
      cameraState.value = CameraState.background;
      _stopStream();
      _cameraController?.dispose();
      _cameraController = null;
    } else if (state == AppLifecycleState.resumed) {
      // Foreground
      debugPrint('[CameraController] App resumed. Re-initializing.');
      _isStopping = false;
      _initializeCamera();
    }
  }

  Future<void> _initializeCamera() async {
    _isStopping = false;
    cameraState.value = CameraState.initializing;
    errorMessage.value = '';

    try {
      // Fetch available cameras
      _cameras = await cam.availableCameras();

      if (_cameras.isEmpty) {
        throw Exception('No cameras found on device');
      }

      int initialIndex = _cameras.indexWhere(
        (c) => c.lensDirection == cam.CameraLensDirection.front,
      );
      if (initialIndex == -1) initialIndex = 0;

      currentCameraIndex.value = initialIndex;
      await _initController(_cameras[initialIndex]);
    } catch (e) {
      debugPrint('[CameraController] Init Error: $e');
      if (e is cam.CameraException && e.code == 'CameraAccessDenied') {
        cameraState.value = CameraState.permissionDenied;
      } else {
        cameraState.value = CameraState.error;
        errorMessage.value = 'Failed to initialize camera: $e';
      }
    }
  }

  Future<void> _initController(cam.CameraDescription camera) async {
    final controller = cam.CameraController(
      camera,
      cam.ResolutionPreset.medium, // 720p is good balance for ML
      enableAudio: false,
      imageFormatGroup: Platform.isAndroid
          ? cam
                .ImageFormatGroup
                .nv21 // NV21 is standard for Android ML Kit
          : cam.ImageFormatGroup.bgra8888, // iOS
    );

    try {
      debugPrint(
        '[CameraController] Found ${_cameras.length} cameras. Initializing ${camera.name} (${camera.lensDirection})',
      );
      await controller.initialize();
      _cameraController = controller;

      currentCameraLens.value =
          camera.lensDirection == cam.CameraLensDirection.front ? 1 : 0;

      cameraState.value = CameraState.ready;
      debugPrint(
        '[CameraController] Camera initialized successfully. State: Ready.',
      );
      _startStream();
    } catch (e) {
      debugPrint('[CameraController] Controller Init Error: $e');
      cameraState.value = CameraState.error;
      errorMessage.value = '$e';
    }
  }

  void switchCamera() async {
    if (_cameras.length < 2) return;

    // Cycle index
    int newIndex = (currentCameraIndex.value + 1) % _cameras.length;
    currentCameraIndex.value = newIndex;

    // Stop current
    _isStopping = true;
    await _stopStream();
    await _cameraController?.dispose();
    _cameraController = null;

    // Init new
    _isStopping = false;
    await _initController(_cameras[newIndex]);
  }

  Future<void> takePicture() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      debugPrint('[CameraController] Cannot take picture: Not initialized');
      return;
    }
    if (_cameraController!.value.isTakingPicture) return;

    try {
      final file = await _cameraController!.takePicture();
      debugPrint('[CameraController] Picture taken: ${file.path}');

      // Stop stream to free resources/prevent background work
      await _stopStream();

      // Navigate to Preview Screen
      await Get.to(
        () => ImagePreviewScreen(
          imagePath: file.path,
          emotion: currentEmotion.value,
        ),
      );

      // Restart stream when coming back
      _startStream();
    } catch (e) {
      debugPrint('[CameraController] Capture Error: $e');
      Get.snackbar("Error", "Failed to capture image");
    }
  }

  // --- Streaming & Processing ---

  Future<void> _startStream() async {
    if (_cameraController == null) return;

    try {
      await _cameraController!.startImageStream(_processCameraImage);
    } catch (e) {
      debugPrint('[CameraController] Start Stream Error: $e');
    }
  }

  Future<void> _stopStream() async {
    _isStopping = true;
    try {
      if (_cameraController != null &&
          _cameraController!.value.isStreamingImages) {
        await _cameraController!.stopImageStream();
      }
    } catch (_) {}
    detectedFaceBounds.value = null;
    isFaceDetected.value = false;
  }

  void _processCameraImage(cam.CameraImage image) {
    if (_isProcessingFrame ||
        _isStopping ||
        _cameraController == null ||
        !_cameraController!.value.isInitialized) {
      return;
    }
    _isProcessingFrame = true;

    final now = DateTime.now();

    // A. Local Face Detection Throttling (Reduce Buffer Pressure & Log Spam)
    bool runFaceDetection = false;
    if (_lastFaceDetectionTime == null ||
        now.difference(_lastFaceDetectionTime!) >= _faceDetectionInterval) {
      runFaceDetection = true;
    }

    // 2. API Sampling (Priority: Background)
    // Check if we need to send a frame to API
    bool runApiSampling = false;

    // Reset emotion if face lost for > 1 second
    if (!isFaceDetected.value) {
      if (_lastFaceDetectedTime != null &&
          now.difference(_lastFaceDetectedTime!).inSeconds > 1) {
        currentEmotion.value = null;
        _lastFaceDetectedTime = null;
      }
    } else {
      // Face is detected
      _lastFaceDetectedTime = now;
    }

    if (_lastApiCallTime == null ||
        now.difference(_lastApiCallTime!) > ApiConfig.samplingInterval) {
      if (isFaceDetected.value &&
          !isAnalyzingEmotion.value &&
          cameraState.value == CameraState.ready) {
        runApiSampling = true;
        _lastApiCallTime = now;
      }
    }

    // Acknowledge processing before taking time-consuming steps
    if (runFaceDetection) {
      _lastFaceDetectionTime = now;
      // print('[CameraController] Throttled frame processing triggered.');
    }

    // Shared "InputImage" creation (if needed for ML Kit)
    InputImage? inputImage;
    if (runFaceDetection) {
      inputImage = _inputImageFromCameraImage(image);
      if (inputImage == null) {
        debugPrint('[CameraController] Warning: InputImage is null');
      }
    }

    // NEW: Copy API data SYNCHRONOUSLY before image is disposed
    CameraImageTransferData? apiTransferData;
    if (runApiSampling) {
      final rotation = Platform.isAndroid ? _getRotationCompensation() : 0;
      apiTransferData = _extractDataFromImageSync(image, rotation);
    }

    Future<void> processingSteps() async {
      // 1. Face Detection
      if (runFaceDetection && inputImage != null) {
        try {
          final startTime = DateTime.now();
          final faces = await _faceDetector.processImage(inputImage);
          final duration = DateTime.now().difference(startTime);

          if (faces.isNotEmpty) {
            faces.sort(
              (a, b) => (b.boundingBox.width * b.boundingBox.height).compareTo(
                a.boundingBox.width * a.boundingBox.height,
              ),
            );
            detectedFaceBounds.value = faces.first.boundingBox;
            isFaceDetected.value = true;
            debugPrint(
              '[FaceDetector] Face detected in ${duration.inMilliseconds}ms. Bounds: ${detectedFaceBounds.value}',
            );
          } else {
            detectedFaceBounds.value = null;
            isFaceDetected.value = false;
            // Only log face loss occasionally to avoid spam
            // print('[FaceDetector] No face detected (${duration.inMilliseconds}ms)');
          }
        } catch (e) {
          debugPrint('[FaceDetector] Error: $e');
        }
      }

      // 2. API Sampling (Now using pre-extracted data)
      if (runApiSampling && apiTransferData != null) {
        _dispatchApiRequest(apiTransferData);
      }
    }

    // Execute
    processingSteps().whenComplete(() {
      _isProcessingFrame = false;
    });
  }

  void _dispatchApiRequest(CameraImageTransferData transferData) async {
    // 1. Drop if busy (Real-time optimization)
    if (_isApiBusy) return;

    _isApiBusy = true;
    isAnalyzingEmotion.value = true;

    try {
      // Send to API Service via Isolate conversion
      final conversionStart = DateTime.now();
      final jpegBytes = await compute(_convertYUV420toJPEG, transferData);
      final conversionDuration = DateTime.now().difference(conversionStart);

      if (jpegBytes != null) {
        debugPrint(
          '[CameraController] YUV to JPEG conversion took ${conversionDuration.inMilliseconds}ms. Dispatching to API...',
        );
        final result = await EmotionApiService().detectEmotionFromCamera(
          imageBytes: jpegBytes,
          skipCompression: true, // Already compressed to JPEG
        );

        // Update Status
        final apiService = EmotionApiService();
        if (apiService.connectionStatus == 'connected') {
          apiStatus.value = 'Connected';
        } else if (apiService.connectionStatus == 'disconnected') {
          apiStatus.value = 'Offline';
        } else {
          apiStatus.value = 'Error';
        }

        if (result != null) {
          currentEmotion.value = result;
          _updateDynamicUI(result);

          // Integrate with Global EmotionController (History)
          try {
            if (Get.isRegistered<EmotionController>()) {
              Get.find<EmotionController>().addEmotion(result);
            }
          } catch (e) {
            debugPrint(
              '[CameraController] Failed to update EmotionController: $e',
            );
          }
        } else {
          debugPrint(
            '[CameraController] API returned null (Low confidence/No face)',
          );
        }
      }
    } catch (e) {
      debugPrint('[CameraController] API Dispatch Error: $e');
      apiStatus.value = 'Error';
    } finally {
      // Release output lock
      _isApiBusy = false;
      isAnalyzingEmotion.value = false;
    }
  }

  void _updateDynamicUI(EmotionModel emotion) {
    switch (emotion.type.toLowerCase()) {
      case 'happy':
        emotionColor.value = const Color(0xFFFFD700); // Gold
        break;
      case 'sad':
        emotionColor.value = const Color(0xFF2196F3); // Blue
        break;
      case 'angry':
        emotionColor.value = const Color(0xFFF44336); // Red
        break;
      case 'surprise':
        emotionColor.value = const Color(0xFFFF9800); // Orange
        break;
      case 'fear':
        emotionColor.value = const Color(0xFF9C27B0); // Purple
        break;
      case 'neutral':
      default:
        emotionColor.value = const Color(0xFF4CAF50); // Green
        break;
    }
  }

  InputImage? _inputImageFromCameraImage(cam.CameraImage image) {
    if (_cameras.isEmpty) return null;
    final camera = _cameras[currentCameraIndex.value];
    final sensorOrientation = camera.sensorOrientation;

    // Determine rotation
    InputImageRotation? rotation;
    if (Platform.isIOS) {
      rotation = InputImageRotationValue.fromRawValue(sensorOrientation);
    } else if (Platform.isAndroid) {
      final rotationCompensation = _getRotationCompensation();
      rotation = InputImageRotationValue.fromRawValue(rotationCompensation);
    }

    if (rotation == null) {
      debugPrint('[CameraController] Warning: Could not determine rotation');
      return null;
    }

    // print('[CameraController] Rotation: $rotation, Format: ${image.format.raw}'); // Debug

    final format = InputImageFormatValue.fromRawValue(image.format.raw);
    if (format == null) return null; // Unsupported format

    return InputImage.fromBytes(
      bytes: _concatenatePlanes(image.planes),
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: format,
        bytesPerRow: image.planes[0].bytesPerRow, // Main plane
      ),
    );
  }

  Uint8List _concatenatePlanes(List<cam.Plane> planes) {
    final WriteBuffer allBytes = WriteBuffer();
    for (final cam.Plane plane in planes) {
      allBytes.putUint8List(plane.bytes);
    }
    return allBytes.done().buffer.asUint8List();
  }

  final Map<DeviceOrientation, int> _orientations = {
    DeviceOrientation.portraitUp: 0,
    DeviceOrientation.landscapeLeft: 90,
    DeviceOrientation.portraitDown: 180,
    DeviceOrientation.landscapeRight: 270,
  };

  int _getRotationCompensation() {
    if (_cameras.isEmpty || _cameraController == null) return 0;

    final camera = _cameras[currentCameraIndex.value];
    final sensorOrientation = camera.sensorOrientation;

    var rotationCompensation =
        _orientations[_cameraController!.value.deviceOrientation];
    if (rotationCompensation == null) return 0;

    if (camera.lensDirection == cam.CameraLensDirection.front) {
      // Front camera
      rotationCompensation = (sensorOrientation + rotationCompensation) % 360;
    } else {
      // Back camera
      rotationCompensation =
          (sensorOrientation - rotationCompensation + 360) % 360;
    }
    return rotationCompensation;
  }

  CameraImageTransferData? _extractDataFromImageSync(
    cam.CameraImage image,
    int rotation,
  ) {
    try {
      final yBytes = Uint8List.fromList(image.planes[0].bytes);
      final uBytes = Uint8List.fromList(image.planes[1].bytes);
      final vBytes = Uint8List.fromList(image.planes[2].bytes);

      return CameraImageTransferData(
        yBytes: yBytes,
        uBytes: uBytes,
        vBytes: vBytes,
        width: image.width,
        height: image.height,
        yStride: image.planes[0].bytesPerRow,
        uvStride: image.planes[1].bytesPerRow,
        uvPixelStride: image.planes[1].bytesPerPixel ?? 1,
        rotation: rotation,
      );
    } catch (e) {
      return null;
    }
  }
}

// --- Helper Functions/Classes ---

class CameraImageTransferData {
  final Uint8List yBytes;
  final Uint8List uBytes;
  final Uint8List vBytes;
  final int width;
  final int height;
  final int yStride;
  final int uvStride;
  final int uvPixelStride;
  final int rotation;

  CameraImageTransferData({
    required this.yBytes,
    required this.uBytes,
    required this.vBytes,
    required this.width,
    required this.height,
    required this.yStride,
    required this.uvStride,
    required this.uvPixelStride,
    this.rotation = 0,
  });
}

// Isolate Function
Uint8List? _convertYUV420toJPEG(CameraImageTransferData data) {
  try {
    // 1. Convert YUV to Image
    // Note: img.Image usually works in RGB.
    // We can use a simpler approach if we don't want to implement YUV conversion manually:
    // Just return generic image if we can't do it.
    // BUT we need it for the API.

    // Minimal YUV conversion (Gray scale is faster and enough for emotion? No, usually color).
    // Let's implement basic YUV -> RGB
    // Ref: https://github.com/brendan-duncan/image/blob/main/lib/src/util/yuv_conversion.dart (conceptually)

    final int w = data.width;
    final int h = data.height;
    final img.Image image = img.Image(width: w, height: h);

    final int yStride = data.yStride;
    final int uvStride = data.uvStride;
    final int uvPixelStride = data.uvPixelStride;

    final Uint8List yp = data.yBytes;
    final Uint8List up = data.uBytes;
    final Uint8List vp = data.vBytes;

    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        final int yIndex = y * yStride + x;
        final int uvIndex = (y ~/ 2) * uvStride + (x ~/ 2) * uvPixelStride;

        // Safety check for indices
        if (yIndex >= yp.length ||
            uvIndex >= up.length ||
            uvIndex >= vp.length) {
          continue;
        }

        int Y = yp[yIndex];
        int U = up[uvIndex] - 128;
        int V = vp[uvIndex] - 128;

        int R = (Y + 1.402 * V).round().clamp(0, 255);
        int G = (Y - 0.344136 * U - 0.714136 * V).round().clamp(0, 255);
        int B = (Y + 1.772 * U).round().clamp(0, 255);

        image.setPixelRgb(x, y, R, G, B);
      }
    }

    // 2. Rotate if needed
    // Use the rotation calculated from device orientation
    img.Image rotated = image;
    if (data.rotation != 0) {
      rotated = img.copyRotate(image, angle: data.rotation);
    }

    // 3. Compress
    return Uint8List.fromList(img.encodeJpg(rotated, quality: 70));
  } catch (e) {
    print('YUV conversion error: $e');
    return null;
  }
}
