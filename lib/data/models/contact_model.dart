import 'package:uuid/uuid.dart';

class ContactModel {
  final String id;
  final String name;
  final String phoneNumber;

  ContactModel({String? id, required this.name, required this.phoneNumber})
    : id = id ?? Uuid().v4();

  factory ContactModel.fromJson(Map<String, dynamic> json) {
    return ContactModel(
      id: json['id'] as String,
      name: json['name'] as String,
      phoneNumber: json['phoneNumber'] as String,
    );
  }

  Map<String, dynamic> toJson() {
    return {'id': id, 'name': name, 'phoneNumber': phoneNumber};
  }
}
