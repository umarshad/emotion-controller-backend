import 'package:get/get.dart';
import '../models/note_model.dart';
import '../services/storage_service.dart';

/// Controller for managing user notes and tasks
class NotesController extends GetxController {
  // Observable list of notes
  final RxList<NoteModel> notes = <NoteModel>[].obs;

  // Loading state
  final RxBool isLoading = false.obs;

  StorageService get _storageService => Get.find<StorageService>();

  @override
  void onInit() {
    super.onInit();
    _loadNotes();
  }

  /// Add a new note
  void addNote(String title, String content) {
    if (title.isEmpty) return;

    final newNote = NoteModel(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      title: title,
      content: content,
      timestamp: DateTime.now(),
    );

    notes.insert(0, newNote);
    _saveNotes();
  }

  /// Update an existing note
  void updateNote(NoteModel updatedNote) {
    final index = notes.indexWhere((n) => n.id == updatedNote.id);
    if (index != -1) {
      notes[index] = updatedNote;
      _saveNotes();
    }
  }

  /// Toggle completion status
  void toggleNoteStatus(String id) {
    final index = notes.indexWhere((n) => n.id == id);
    if (index != -1) {
      notes[index] = notes[index].copyWith(
        isCompleted: !notes[index].isCompleted,
      );
      _saveNotes();
    }
  }

  /// Delete a note
  void deleteNote(String id) {
    notes.removeWhere((n) => n.id == id);
    _saveNotes();
  }

  /// Clear all notes
  void clearNotes() {
    notes.clear();
    _storageService.clearNotes();
  }

  /// Load notes from storage
  void _loadNotes() {
    final loadedNotes = _storageService.loadNotes();
    if (loadedNotes.isNotEmpty) {
      notes.assignAll(loadedNotes);
    }
  }

  /// Save notes to storage
  void _saveNotes() {
    _storageService.saveNotes(notes.toList());
  }
}
