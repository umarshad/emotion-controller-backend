import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import '../../data/constants/colors.dart';
import '../../data/controllers/notes_controller.dart';
import '../../data/models/note_model.dart';
import '../../widgets/custom_text.dart';

class AddEditNoteScreen extends StatefulWidget {
  final NoteModel? note;
  const AddEditNoteScreen({super.key, this.note});

  @override
  State<AddEditNoteScreen> createState() => _AddEditNoteScreenState();
}

class _AddEditNoteScreenState extends State<AddEditNoteScreen> {
  late TextEditingController _titleController;
  late TextEditingController _contentController;
  final _formKey = GlobalKey<FormState>();

  @override
  void initState() {
    super.initState();
    _titleController = TextEditingController(text: widget.note?.title ?? '');
    _contentController = TextEditingController(
      text: widget.note?.content ?? '',
    );
  }

  @override
  void dispose() {
    _titleController.dispose();
    _contentController.dispose();
    super.dispose();
  }

  void _saveNote() {
    if (_formKey.currentState!.validate()) {
      final controller = Get.find<NotesController>();
      if (widget.note == null) {
        controller.addNote(_titleController.text, _contentController.text);
      } else {
        final updatedNote = widget.note!.copyWith(
          title: _titleController.text,
          content: _contentController.text,
          timestamp: DateTime.now(),
        );
        controller.updateNote(updatedNote);
      }
      Get.back();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.lightBackground,
      appBar: AppBar(
        title: Text(widget.note == null ? 'New Note' : 'Edit Note'),
        actions: [
          TextButton(
            onPressed: _saveNote,
            child: CustomText(
              'Save',
              color: AppColors.primary,
              fontWeight: FontWeight.bold,
              fontSize: 16,
            ),
          ),
          SizedBox(width: 8.w),
        ],
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(24.w),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              CustomText(
                'Title',
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: AppColors.textSecondary,
              ),
              SizedBox(height: 8.h),
              TextFormField(
                controller: _titleController,
                style: TextStyle(
                  fontSize: 18.sp,
                  fontWeight: FontWeight.bold,
                  color: AppColors.textPrimary,
                ),
                decoration: InputDecoration(
                  hintText: 'Enter title...',
                  hintStyle: TextStyle(
                    color: AppColors.textSecondary.withValues(alpha: 0.5),
                  ),
                  contentPadding: EdgeInsets.symmetric(
                    horizontal: 16.w,
                    vertical: 12.h,
                  ),
                ),
                validator: (value) =>
                    value == null || value.isEmpty ? 'Title is required' : null,
              ),
              SizedBox(height: 24.h),
              CustomText(
                'Detailed notes (Optional)',
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: AppColors.textSecondary,
              ),
              SizedBox(height: 8.h),
              TextFormField(
                controller: _contentController,
                maxLines: 15,
                style: TextStyle(
                  fontSize: 16.sp,
                  color: AppColors.textPrimary,
                  height: 1.5,
                ),
                decoration: InputDecoration(
                  hintText: 'Write something here...',
                  hintStyle: TextStyle(
                    color: AppColors.textSecondary.withValues(alpha: 0.5),
                  ),
                  contentPadding: EdgeInsets.all(16.w),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
