import 'package:get/get.dart';
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/contact_model.dart';

class ContactController extends GetxController {
  final RxList<ContactModel> contacts = <ContactModel>[].obs;
  static const String _storageKey = 'trusted_contacts';

  @override
  void onInit() {
    super.onInit();
    loadContacts();
  }

  Future<void> loadContacts() async {
    final prefs = await SharedPreferences.getInstance();
    final String? contactsJson = prefs.getString(_storageKey);
    if (contactsJson != null) {
      final List<dynamic> decoded = jsonDecode(contactsJson);
      contacts.assignAll(
        decoded.map((json) => ContactModel.fromJson(json)).toList(),
      );
    }
  }

  Future<void> saveContacts() async {
    final prefs = await SharedPreferences.getInstance();
    final String encoded = jsonEncode(contacts.map((c) => c.toJson()).toList());
    await prefs.setString(_storageKey, encoded);
  }

  void addContact(String name, String phoneNumber) {
    contacts.add(ContactModel(name: name, phoneNumber: phoneNumber));
    saveContacts();
  }

  void deleteContact(String id) {
    contacts.removeWhere((c) => c.id == id);
    saveContacts();
  }
}
