import 'package:get/get.dart';
import '../models/emotion_model.dart';
import '../models/journal_model.dart';
import '../services/storage_service.dart';

/// Controller for managing emotion detection and history
class EmotionController extends GetxController {
  // Observable list of detected emotions
  final RxList<EmotionModel> emotions = <EmotionModel>[].obs;

  // Observable list of journal entries
  final RxList<JournalModel> journalEntries = <JournalModel>[].obs;

  // Current emotion being displayed
  final Rx<EmotionModel?> currentEmotion = Rx<EmotionModel?>(null);

  // Loading state
  final RxBool isLoading = false.obs;

  StorageService get _storageService => Get.find<StorageService>();

  @override
  void onInit() {
    super.onInit();
    // Load persisted data
    _loadData();
  }

  /// Add a new detected emotion
  void addEmotion(EmotionModel emotion) {
    emotions.insert(0, emotion); // Add to beginning for chronological order
    currentEmotion.value = emotion;
    _saveEmotions();
  }

  /// Update user rating for an emotion
  void updateEmotionRating(String id, int rating) {
    final index = emotions.indexWhere((e) => e.id == id);
    if (index != -1) {
      emotions[index] = emotions[index].copyWith(userRating: rating);
      if (currentEmotion.value?.id == id) {
        currentEmotion.value = emotions[index];
      }
      _saveEmotions();
    }
  }

  /// Get recent emotions (last 5)
  List<EmotionModel> getRecentEmotions({int count = 5}) {
    return emotions.take(count).toList();
  }

  /// Get emotions by type
  List<EmotionModel> getEmotionsByType(String type) {
    return emotions
        .where((e) => e.type.toLowerCase() == type.toLowerCase())
        .toList();
  }

  /// Get emotions by detection method
  List<EmotionModel> getEmotionsByMethod(String method) {
    return emotions.where((e) => e.detectionMethod == method).toList();
  }

  /// Get emotion statistics
  Map<String, int> getEmotionStats() {
    final stats = <String, int>{};
    for (final emotion in emotions) {
      stats[emotion.type] = (stats[emotion.type] ?? 0) + 1;
    }
    return stats;
  }

  /// Clear all emotions
  void clearEmotions() {
    emotions.clear();
    currentEmotion.value = null;
    _storageService.clearEmotions();
  }

  /// Delete specific emotion
  void deleteEmotion(EmotionModel emotion) {
    emotions.remove(emotion);
    if (currentEmotion.value == emotion) {
      currentEmotion.value = null;
    }
    _saveEmotions();

    // Also delete associated journal entries
    journalEntries.removeWhere((j) => j.emotionId == emotion.id);
    _saveJournals();
  }

  /// Add a journal entry
  void addJournalEntry(JournalModel entry) {
    journalEntries.insert(0, entry);
    _saveJournals();
  }

  /// Get journal entry for a specific emotion
  JournalModel? getJournalForEmotion(String emotionId) {
    try {
      return journalEntries.firstWhere((j) => j.emotionId == emotionId);
    } catch (e) {
      return null;
    }
  }

  /// Delete journal entry
  void deleteJournalEntry(String entryId) {
    journalEntries.removeWhere((j) => j.id == entryId);
    _saveJournals();
  }

  /// Refresh statistics (triggers reactive updates)
  void refreshStats() {
    // Trigger reactive update by reassigning
    final current = List<EmotionModel>.from(emotions);
    emotions.clear();
    emotions.addAll(current);
  }

  /// Load data from storage
  void _loadData() {
    // Load emotions
    final loadedEmotions = _storageService.loadEmotions();
    if (loadedEmotions.isNotEmpty) {
      emotions.assignAll(loadedEmotions);
      if (emotions.isNotEmpty) {
        currentEmotion.value = emotions.first;
      }
    }

    // Load journals
    final loadedJournals = _storageService.loadJournalEntries();
    if (loadedJournals.isNotEmpty) {
      journalEntries.assignAll(loadedJournals);
    }
  }

  /// Save emotions to storage
  void _saveEmotions() {
    _storageService.saveEmotions(emotions.toList());
  }

  /// Save journals to storage
  void _saveJournals() {
    _storageService.saveJournalEntries(journalEntries.toList());
  }
}
