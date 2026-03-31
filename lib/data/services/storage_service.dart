import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:get/get.dart';
import 'package:shared_preferences/shared_preferences.dart' as sp;
import '../models/emotion_model.dart';
import '../models/journal_model.dart';
import '../models/note_model.dart';

class StorageService extends GetxService {
  sp.SharedPreferences? _prefs;
  static const String _keyEmotions = 'emotions_default';
  static const String _keyJournals = 'journals_default';
  static const String _keyNotes = 'notes_default';

  Future<StorageService> init() async {
    try {
      _prefs = await sp.SharedPreferences.getInstance();
    } catch (e) {
      debugPrint(
        '[StorageService] Critical Error: Failed to initialize SharedPreferences: $e',
      );
      // App continues without persistence
    }
    return this;
  }

  /// Save list of emotions to local storage
  Future<void> saveEmotions(List<EmotionModel> emotions) async {
    if (_prefs == null) {
      debugPrint(
        '[StorageService] Warning: Storage not initialized. Cannot save.',
      );
      return;
    }

    try {
      final List<String> jsonList = emotions
          .map((e) => jsonEncode(e.toJson()))
          .toList();
      await _prefs!.setStringList(_keyEmotions, jsonList);
    } catch (e) {
      debugPrint('[StorageService] Error saving emotions: $e');
    }
  }

  /// Load list of emotions from local storage
  List<EmotionModel> loadEmotions() {
    if (_prefs == null) {
      debugPrint(
        '[StorageService] Warning: Storage not initialized. Returning empty list.',
      );
      return [];
    }

    try {
      final List<String>? jsonList = _prefs!.getStringList(_keyEmotions);
      if (jsonList == null) return [];

      return jsonList.map((e) => EmotionModel.fromJson(jsonDecode(e))).toList();
    } catch (e) {
      debugPrint('[StorageService] Error loading emotions: $e');
      return [];
    }
  }

  /// Save list of journal entries to local storage
  Future<void> saveJournalEntries(List<JournalModel> entries) async {
    if (_prefs == null) return;

    try {
      final List<String> jsonList = entries
          .map((e) => jsonEncode(e.toJson()))
          .toList();
      await _prefs!.setStringList(_keyJournals, jsonList);
    } catch (e) {
      debugPrint('[StorageService] Error saving journals: $e');
    }
  }

  /// Load list of journal entries from local storage
  List<JournalModel> loadJournalEntries() {
    if (_prefs == null) return [];

    try {
      final List<String>? jsonList = _prefs!.getStringList(_keyJournals);
      if (jsonList == null) return [];

      return jsonList.map((e) => JournalModel.fromJson(jsonDecode(e))).toList();
    } catch (e) {
      debugPrint('[StorageService] Error loading journals: $e');
      return [];
    }
  }

  /// Save list of notes to local storage
  Future<void> saveNotes(List<NoteModel> notes) async {
    if (_prefs == null) return;

    try {
      final List<String> jsonList = notes
          .map((n) => jsonEncode(n.toJson()))
          .toList();
      await _prefs!.setStringList(_keyNotes, jsonList);
    } catch (e) {
      debugPrint('[StorageService] Error saving notes: $e');
    }
  }

  /// Load list of notes from local storage
  List<NoteModel> loadNotes() {
    if (_prefs == null) return [];

    try {
      final List<String>? jsonList = _prefs!.getStringList(_keyNotes);
      if (jsonList == null) return [];

      return jsonList.map((n) => NoteModel.fromJson(jsonDecode(n))).toList();
    } catch (e) {
      debugPrint('[StorageService] Error loading notes: $e');
      return [];
    }
  }

  /// Clear all saved emotions
  Future<void> clearEmotions() async {
    if (_prefs == null) return;

    try {
      await _prefs!.remove(_keyEmotions);
    } catch (e) {
      debugPrint('[StorageService] Error clearing emotions: $e');
    }
  }

  /// Clear all journal entries
  Future<void> clearJournalEntries() async {
    if (_prefs == null) return;

    try {
      await _prefs!.remove(_keyJournals);
    } catch (e) {
      debugPrint('[StorageService] Error clearing journals: $e');
    }
  }

  /// Clear all notes
  Future<void> clearNotes() async {
    if (_prefs == null) return;

    try {
      await _prefs!.remove(_keyNotes);
    } catch (e) {
      debugPrint('[StorageService] Error clearing notes: $e');
    }
  }
}
