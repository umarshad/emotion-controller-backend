import 'package:get/get.dart';

/// Controller for managing bottom navigation state
class NavigationController extends GetxController {
  // Current selected index
  final RxInt currentIndex = 0.obs;

  // Navigation history for back button handling
  final RxList<int> navigationHistory = <int>[0].obs;

  /// Change navigation index
  void changeIndex(int index) {
    if (index == currentIndex.value) return;

    // Add to history if not camera button (index 2)
    if (index != 2) {
      navigationHistory.add(index);
      if (navigationHistory.length > 10) {
        navigationHistory.removeAt(0);
      }
    }

    currentIndex.value = index;
  }

  /// Navigate to camera (special handling for center button)
  void navigateToCamera() {
    changeIndex(2);
  }

  /// Get current screen index
  int getCurrentIndex() => currentIndex.value;

  /// Check if camera is active
  bool isCameraActive() => currentIndex.value == 2;
}
