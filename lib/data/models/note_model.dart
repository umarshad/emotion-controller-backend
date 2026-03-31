/// Note model for user tasks and reflections
class NoteModel {
  final String id;
  final String title;
  final String content;
  final DateTime timestamp;
  final bool isCompleted;

  NoteModel({
    required this.id,
    required this.title,
    required this.content,
    required this.timestamp,
    this.isCompleted = false,
  });

  factory NoteModel.fromJson(Map<String, dynamic> json) {
    return NoteModel(
      id: json['id'] as String,
      title: json['title'] as String,
      content: json['content'] as String,
      timestamp: DateTime.parse(json['timestamp'] as String),
      isCompleted: json['isCompleted'] as bool? ?? false,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'title': title,
      'content': content,
      'timestamp': timestamp.toIso8601String(),
      'isCompleted': isCompleted,
    };
  }

  NoteModel copyWith({
    String? id,
    String? title,
    String? content,
    DateTime? timestamp,
    bool? isCompleted,
  }) {
    return NoteModel(
      id: id ?? this.id,
      title: title ?? this.title,
      content: content ?? this.content,
      timestamp: timestamp ?? this.timestamp,
      isCompleted: isCompleted ?? this.isCompleted,
    );
  }
}
