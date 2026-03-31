import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import '../../data/constants/colors.dart';
import '../../data/controllers/notes_controller.dart';
import '../../data/models/note_model.dart';
import '../../widgets/custom_text.dart';
import 'add_edit_note_screen.dart';
import 'package:intl/intl.dart';

class NotesScreen extends StatelessWidget {
  const NotesScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final controller = Get.find<NotesController>();

    return Scaffold(
      backgroundColor: AppColors.lightBackground,
      appBar: AppBar(
        title: const Text('My Notes'),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.delete_sweep_outlined),
            onPressed: () => _showClearAllDialog(context, controller),
          ),
        ],
      ),
      body: Obx(() {
        if (controller.notes.isEmpty) {
          return _buildEmptyState();
        }

        return ListView.builder(
          padding: EdgeInsets.all(20.w),
          itemCount: controller.notes.length,
          itemBuilder: (context, index) {
            final note = controller.notes[index];
            return _buildNoteCard(context, note, controller);
          },
        );
      }),
      floatingActionButton: FloatingActionButton(
        onPressed: () => Get.to(() => const AddEditNoteScreen()),
        backgroundColor: AppColors.primary,
        child: const Icon(Icons.add, color: Colors.white),
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.note_alt_outlined,
            size: 80.sp,
            color: AppColors.textSecondary.withValues(alpha: 0.3),
          ),
          SizedBox(height: 16.h),
          CustomText(
            'No notes yet',
            fontSize: 18,
            fontWeight: FontWeight.w600,
            color: AppColors.textSecondary,
          ),
          SizedBox(height: 8.h),
          CustomText(
            'Tap + to add your first note or task',
            fontSize: 14,
            color: AppColors.textSecondary,
          ),
        ],
      ),
    );
  }

  Widget _buildNoteCard(
    BuildContext context,
    NoteModel note,
    NotesController controller,
  ) {
    return Container(
      margin: EdgeInsets.only(bottom: 16.h),
      decoration: BoxDecoration(
        color: AppColors.lightCardBackground,
        borderRadius: BorderRadius.circular(16.r),
        border: Border.all(color: AppColors.lightBorder),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.02),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(16.r),
          onTap: () => Get.to(() => AddEditNoteScreen(note: note)),
          child: Padding(
            padding: EdgeInsets.all(16.w),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: CustomText(
                        note.title,
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: AppColors.textPrimary,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                    Checkbox(
                      value: note.isCompleted,
                      activeColor: AppColors.primary,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(4.r),
                      ),
                      onChanged: (_) => controller.toggleNoteStatus(note.id),
                    ),
                  ],
                ),
                if (note.content.isNotEmpty) ...[
                  SizedBox(height: 8.h),
                  CustomText(
                    note.content,
                    fontSize: 14,
                    color: AppColors.textSecondary,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
                SizedBox(height: 12.h),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    CustomText(
                      DateFormat(
                        'MMM dd, yyyy • hh:mm a',
                      ).format(note.timestamp),
                      fontSize: 12,
                      color: AppColors.textSecondary.withValues(alpha: 0.6),
                    ),
                    IconButton(
                      icon: Icon(
                        Icons.delete_outline,
                        size: 20.sp,
                        color: Colors.redAccent,
                      ),
                      onPressed: () => controller.deleteNote(note.id),
                      padding: EdgeInsets.zero,
                      constraints: const BoxConstraints(),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  void _showClearAllDialog(BuildContext context, NotesController controller) {
    Get.dialog(
      AlertDialog(
        title: const Text('Clear All Notes'),
        content: const Text(
          'Are you sure you want to delete all notes? This action cannot be undone.',
        ),
        actions: [
          TextButton(onPressed: () => Get.back(), child: const Text('Cancel')),
          TextButton(
            onPressed: () {
              controller.clearNotes();
              Get.back();
            },
            child: const Text('Clear All', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
  }
}
