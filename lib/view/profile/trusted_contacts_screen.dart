import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import 'package:url_launcher/url_launcher.dart';
import '../../data/controllers/contact_controller.dart';
import '../../widgets/custom_text.dart';
import '../../widgets/custom_app_bar.dart';
import '../../data/constants/colors.dart';

class TrustedContactsScreen extends StatelessWidget {
  const TrustedContactsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final controller = Get.put(ContactController());
    final nameController = TextEditingController();
    final phoneController = TextEditingController();

    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      appBar: const CustomAppBar(title: "Loved Ones", showBackButton: true),
      body: Obx(
        () => ListView.separated(
          padding: EdgeInsets.fromLTRB(16.w, 16.h, 16.w, 100.h),
          itemCount: controller.contacts.length,
          separatorBuilder: (context, index) => SizedBox(height: 12.h),
          itemBuilder: (context, index) {
            final contact = controller.contacts[index];
            return Container(
              padding: EdgeInsets.all(16.w),
              decoration: BoxDecoration(
                color: Theme.of(context).cardColor,
                borderRadius: BorderRadius.circular(16.r),
              ),
              child: Row(
                children: [
                  CircleAvatar(
                    backgroundColor: AppColors.primary.withValues(alpha: 0.1),
                    child: CustomText(
                      contact.name[0].toUpperCase(),
                      color: AppColors.primary,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(width: 16.w),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        CustomText(
                          contact.name,
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                        ),
                        CustomText(
                          contact.phoneNumber,
                          color: AppColors.textLight,
                          fontSize: 14,
                        ),
                      ],
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.call, color: Colors.green),
                    onPressed: () => _callNumber(contact.phoneNumber),
                  ),
                  IconButton(
                    icon: const Icon(Icons.delete_outline, color: Colors.red),
                    onPressed: () => controller.deleteContact(contact.id),
                  ),
                ],
              ),
            );
          },
        ),
      ),
      floatingActionButton: FloatingActionButton.extended(
        backgroundColor: AppColors.primary,
        onPressed: () => _showAddContactBottomSheet(
          context,
          controller,
          nameController,
          phoneController,
        ),
        label: const Text("Add Loved One"),
        icon: const Icon(Icons.add),
      ),
    );
  }

  void _showAddContactBottomSheet(
    BuildContext context,
    ContactController controller,
    TextEditingController nameCtrl,
    TextEditingController phoneCtrl,
  ) {
    Get.bottomSheet(
      Container(
        padding: EdgeInsets.all(32.w),
        decoration: BoxDecoration(
          color: Theme.of(context).cardColor,
          borderRadius: BorderRadius.vertical(top: Radius.circular(32.r)),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.1),
              blurRadius: 20,
              offset: const Offset(0, -5),
            ),
          ],
        ),
        child: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  CustomText(
                    "Add Loved One",
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                  ),
                  IconButton(
                    onPressed: () => Get.back(),
                    icon: const Icon(Icons.close_rounded),
                  ),
                ],
              ),
              SizedBox(height: 8.h),
              CustomText(
                "Keep your support system close",
                color: AppColors.textSecondary,
                fontSize: 14,
              ),
              SizedBox(height: 32.h),
              _buildStyledTextField(
                controller: nameCtrl,
                label: "Name",
                icon: Icons.person_outline_rounded,
                hint: "e.g. John Doe",
              ),
              SizedBox(height: 20.h),
              _buildStyledTextField(
                controller: phoneCtrl,
                label: "Phone Number",
                icon: Icons.phone_android_rounded,
                hint: "+1 234 567 890",
                keyboardType: TextInputType.phone,
              ),
              SizedBox(height: 40.h),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: () {
                    if (nameCtrl.text.isNotEmpty && phoneCtrl.text.isNotEmpty) {
                      controller.addContact(nameCtrl.text, phoneCtrl.text);
                      nameCtrl.clear();
                      phoneCtrl.clear();
                      Get.back();
                    }
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.primary,
                    foregroundColor: Colors.white,
                    padding: EdgeInsets.symmetric(vertical: 18.h),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16.r),
                    ),
                    elevation: 0,
                  ),
                  child: const Text(
                    "Save Contact",
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                  ),
                ),
              ),
              SizedBox(height: 16.h),
            ],
          ),
        ),
      ),
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      enterBottomSheetDuration: const Duration(milliseconds: 300),
      exitBottomSheetDuration: const Duration(milliseconds: 200),
    );
  }

  Widget _buildStyledTextField({
    required TextEditingController controller,
    required String label,
    required IconData icon,
    required String hint,
    TextInputType keyboardType = TextInputType.text,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        CustomText(
          label,
          fontSize: 14,
          fontWeight: FontWeight.w600,
          color: AppColors.textPrimary,
        ),
        SizedBox(height: 8.h),
        TextField(
          controller: controller,
          keyboardType: keyboardType,
          style: const TextStyle(fontWeight: FontWeight.w500),
          decoration: InputDecoration(
            prefixIcon: Icon(icon, color: AppColors.primary, size: 20),
            hintText: hint,
            hintStyle: TextStyle(color: AppColors.textLight, fontSize: 14),
            filled: true,
            fillColor: AppColors.primary.withValues(alpha: 0.05),
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(16.r),
              borderSide: BorderSide.none,
            ),
            contentPadding: EdgeInsets.symmetric(
              horizontal: 20.w,
              vertical: 16.h,
            ),
          ),
        ),
      ],
    );
  }

  Future<void> _callNumber(String number) async {
    try {
      final uri = Uri.parse("tel:$number");
      if (await canLaunchUrl(uri)) {
        await launchUrl(uri);
      } else {
        Get.snackbar(
          "Call Error",
          "Could not initiate call. Please check your phone settings.",
          snackPosition: SnackPosition.BOTTOM,
          backgroundColor: Colors.redAccent.withValues(alpha: 0.8),
          colorText: Colors.white,
        );
      }
    } catch (e) {
      String message = "An unexpected error occurred: $e";
      if (e.toString().contains("channel-error")) {
        message =
            "Plugin sync error: Please stop the app, run 'flutter clean', and start it again.";
      }
      Get.snackbar(
        "Linking Error",
        message,
        snackPosition: SnackPosition.BOTTOM,
        backgroundColor: Colors.redAccent.withValues(alpha: 0.9),
        colorText: Colors.white,
        duration: const Duration(seconds: 5),
      );
    }
  }
}
