import 'dart:ui';
import 'package:camera/camera.dart' as cam;
import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import '../../data/controllers/camera_controller.dart';
import '../../data/constants/colors.dart';
import '../../widgets/custom_text.dart';
import 'quotes_detail_screen.dart';
import '../profile/trusted_contacts_screen.dart';

class CameraScreen extends StatelessWidget {
  const CameraScreen({super.key});

  @override
  Widget build(BuildContext context) {
    // Inject controller if not already (or find existing)
    // The previous implementation used Get.find because it was likely injected by bindings.
    // Let's use Get.put to be safe, or Get.lazyPut in bindings.
    // Since we overwrote the controller, let's assume bindings might need update or we just put it here.
    final controller = Get.put(CameraController());

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // 1. Camera Preview Layer
          Obx(() {
            if (controller.cameraState.value == CameraState.initializing) {
              return const Center(
                child: CircularProgressIndicator(color: AppColors.primary),
              );
            } else if (controller.cameraState.value ==
                CameraState.permissionDenied) {
              return _buildPermissionDeniedView();
            } else if (controller.cameraState.value == CameraState.error) {
              return _buildErrorView(controller.errorMessage.value);
            } else if (controller.cameraState.value == CameraState.ready &&
                controller.cameraController != null &&
                controller.cameraController!.value.isInitialized) {
              return _buildCameraPreview(controller);
            }
            return const SizedBox.shrink();
          }),

          // 2. Face Detection Overlay Layer
          Obx(() {
            if (controller.cameraState.value != CameraState.ready) {
              return const SizedBox.shrink();
            }

            final rect = controller.detectedFaceBounds.value;
            if (rect == null) return const SizedBox.shrink();

            return _buildFaceOverlay(rect, controller);
          }),

          // 3. UI Controls Layer (Glassmorphism)
          Positioned(
            bottom: 30.h,
            left: 0,
            right: 0,
            child: _buildBottomControls(controller),
          ),

          // API Status Pill (Top Right)
          Positioned(
            top: 40.h,
            right: 20.w,
            child: _buildApiStatusPill(controller),
          ),

          // Top Bar (Removed)
        ],
      ),
    );
  }

  Widget _buildCameraPreview(CameraController controller) {
    if (controller.cameraController == null ||
        !controller.cameraController!.value.isInitialized) {
      return const SizedBox.shrink();
    }

    final camera = controller.cameraController!.value;
    final size = MediaQuery.of(Get.context!).size;

    // Scale to cover screen (full screen)
    // Camera preview is often 4:3 or 16:9. Screen is ~20:9.
    var scale = size.aspectRatio * camera.aspectRatio;
    if (scale < 1) scale = 1 / scale;

    return Transform.scale(
      scale: scale,
      child: Center(child: cam.CameraPreview(controller.cameraController!)),
    );
  }

  Widget _buildFaceOverlay(Rect faceRect, CameraController controller) {
    return LayoutBuilder(
      builder: (context, constraints) {
        if (controller.cameraController == null) return const SizedBox.shrink();

        final previewSize = controller.cameraController!.value.previewSize;
        if (previewSize == null) return const SizedBox.shrink();

        // Portrait mode means mapped width/height are swapped relative to landscape image
        final double videoWidth = previewSize.height;
        final double videoHeight = previewSize.width;

        final double screenWidth = constraints.maxWidth;
        final double screenHeight = constraints.maxHeight;

        // Calculate scale tailored to "Cover" fit
        final double scaleX = screenWidth / videoWidth;
        final double scaleY = screenHeight / videoHeight;
        final double scale = scaleX > scaleY ? scaleX : scaleY;

        // Calculate offset to center
        final double offsetX = (screenWidth - videoWidth * scale) / 2;
        final double offsetY = (screenHeight - videoHeight * scale) / 2;

        // Apply transform
        final double finalX = faceRect.left * scale + offsetX;
        final double finalY = faceRect.top * scale + offsetY;
        final double finalW = faceRect.width * scale;
        final double finalH = faceRect.height * scale;

        // Mirroring for Front Camera
        double drawX = finalX;
        if (controller.currentCameraLens.value == 1) {
          drawX = screenWidth - finalX - finalW; // Mirror
        }

        return Stack(
          children: [
            Positioned(
              left: drawX,
              top: finalY,
              width: finalW,
              height: finalH,
              child: _buildFaceBox(controller),
            ),
          ],
        );
      },
    );
  }

  Widget _buildFaceBox(CameraController controller) {
    return Obx(
      () => GestureDetector(
        onTap: () {
          if (controller.currentEmotion.value != null) {
            Get.to(
              () =>
                  QuotesDetailScreen(emotion: controller.currentEmotion.value!),
            );
          }
        },
        child: Container(
          decoration: BoxDecoration(
            border: Border.all(
              color: controller.emotionColor.value, // Dynamic Color
              width: 3.w,
            ),
            borderRadius: BorderRadius.circular(12.r),
            boxShadow: [
              BoxShadow(
                color: controller.emotionColor.value.withOpacity(0.3),
                blurRadius: 10,
                spreadRadius: 2,
              ),
            ],
          ),
          child: Stack(
            clipBehavior: Clip.none,
            alignment: Alignment.topCenter,
            children: [
              // Emotion Tag
              Positioned(top: -30.h, child: _buildEmotionTag(controller)),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildEmotionTag(CameraController controller) {
    return Obx(() {
      final emotion = controller.currentEmotion.value;
      final isAnalyzing = controller.isAnalyzingEmotion.value;

      String text = "Detecting...";
      Color color = controller.emotionColor.value;

      if (emotion != null) {
        text = emotion.type.toUpperCase();
        if (text == "SAD") {
          // Special handling to encourage reaching out
          return GestureDetector(
            onTap: () => Get.to(() => const TrustedContactsScreen()),
            child: _buildStandardTag(text, color, isAnalyzing, true),
          );
        }
      } else if (isAnalyzing) {
        text = "Analyzing...";
      } else if (controller.apiStatus.value == 'Error') {
        text = "API Error";
        color = Colors.red;
      } else if (controller.apiStatus.value == 'Offline') {
        text = "Offline";
        color = Colors.grey;
      }

      return _buildStandardTag(text, color, isAnalyzing, false);
    });
  }

  Widget _buildStandardTag(
    String text,
    Color color,
    bool isAnalyzing,
    bool pulse,
  ) {
    return Container(
      padding: EdgeInsets.symmetric(horizontal: 12.w, vertical: 4.h),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(16.r),
        border: Border.all(color: color.withOpacity(0.8)),
        boxShadow: [
          BoxShadow(
            color: Colors.black26,
            blurRadius: pulse ? 10 : 4,
            spreadRadius: pulse ? 2 : 0,
          ),
        ],
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (isAnalyzing)
            Padding(
              padding: EdgeInsets.only(right: 6.w),
              child: SizedBox(
                width: 10.w,
                height: 10.w,
                child: CircularProgressIndicator(strokeWidth: 2, color: color),
              ),
            ),
          CustomText(
            pulse ? "$text (Reach Out)" : text,
            color: Colors.white,
            fontSize: 12,
            fontWeight: FontWeight.bold,
          ),
        ],
      ),
    );
  }

  Widget _buildBottomControls(CameraController controller) {
    return ClipRRect(
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
        child: Container(
          height: 120.h,
          color: Colors.transparent, // Transparent background
          padding: EdgeInsets.symmetric(horizontal: 32.w),
          child: Row(
            mainAxisAlignment:
                MainAxisAlignment.center, // Center the capture button
            children: [
              // Gallery (Removed)

              // Capture Button
              GestureDetector(
                onTap: controller.takePicture,
                child: Container(
                  width: 72.w,
                  height: 72.w,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    border: Border.all(color: Colors.white, width: 4.w),
                    color: controller.isAnalyzingEmotion.value
                        ? AppColors.primary.withValues(alpha: 0.1)
                        : Colors.white.withValues(alpha: 0.1),
                  ),
                  child: Center(
                    child: Container(
                      width: 60.w,
                      height: 60.w,
                      decoration: const BoxDecoration(
                        shape: BoxShape.circle,
                        color: Colors.white,
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildApiStatusPill(CameraController controller) {
    return Obx(() {
      final status = controller.apiStatus.value;
      Color color;
      switch (status) {
        case 'Connected':
          color = Colors.greenAccent;
          break;
        case 'Offline':
          color = Colors.grey;
          break;
        case 'Timeout':
          color = Colors.orangeAccent;
          break;
        case 'Error':
          color = Colors.redAccent;
          break;
        default:
          color = Colors.yellowAccent;
      }

      return Container(
        padding: EdgeInsets.symmetric(horizontal: 12.w, vertical: 6.h),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.4),
          borderRadius: BorderRadius.circular(12.r),
          border: Border.all(color: color.withOpacity(0.5), width: 1),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 8.w,
              height: 8.w,
              decoration: BoxDecoration(
                color: color,
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(
                    color: color.withOpacity(0.6),
                    blurRadius: 6,
                    spreadRadius: 1,
                  ),
                ],
              ),
            ),
            SizedBox(width: 8.w),
            Text(
              "API: $status",
              style: TextStyle(
                color: Colors.white,
                fontSize: 12.sp,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      );
    });
  }

  Widget _buildPermissionDeniedView() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.videocam_off, size: 64.sp, color: Colors.grey),
          SizedBox(height: 16.h),
          CustomText(
            "Camera Permission Denied",
            color: Colors.white,
            fontSize: 18,
          ),
          SizedBox(height: 24.h),
          ElevatedButton(
            onPressed: () {
              // Open settings
              // openAppSettings();
            },
            child: const Text("Open Settings"),
          ),
        ],
      ),
    );
  }

  Widget _buildErrorView(String error) {
    return Center(
      child: Padding(
        padding: EdgeInsets.all(16.w),
        child: CustomText(
          "Error: $error",
          color: Colors.red,
          textAlign: TextAlign.center,
        ),
      ),
    );
  }
}
