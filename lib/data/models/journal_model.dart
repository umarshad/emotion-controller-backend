/// Journal entry model
class JournalModel {
  final String id;
  final String emotionId;
  final String prompt;
  final String content;
  final DateTime timestamp;

  JournalModel({
    required this.id,
    required this.emotionId,
    required this.prompt,
    required this.content,
    required this.timestamp,
  });

  factory JournalModel.fromJson(Map<String, dynamic> json) {
    return JournalModel(
      id: json['id'] as String,
      emotionId: json['emotionId'] as String,
      prompt: json['prompt'] as String,
      content: json['content'] as String,
      timestamp: DateTime.parse(json['timestamp'] as String),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'emotionId': emotionId,
      'prompt': prompt,
      'content': content,
      'timestamp': timestamp.toIso8601String(),
    };
  }

  JournalModel copyWith({
    String? id,
    String? emotionId,
    String? prompt,
    String? content,
    DateTime? timestamp,
  }) {
    return JournalModel(
      id: id ?? this.id,
      emotionId: emotionId ?? this.emotionId,
      prompt: prompt ?? this.prompt,
      content: content ?? this.content,
      timestamp: timestamp ?? this.timestamp,
    );
  }
}
