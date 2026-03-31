/// User profile model
class UserProfile {
  final String name;
  final String? email;
  final String? avatarUrl;
  final DateTime? joinDate;

  UserProfile({
    required this.name,
    this.email,
    this.avatarUrl,
    this.joinDate,
  });

  factory UserProfile.defaultUser() {
    return UserProfile(
      name: 'User',
      joinDate: DateTime.now(),
    );
  }

  UserProfile copyWith({
    String? name,
    String? email,
    String? avatarUrl,
    DateTime? joinDate,
  }) {
    return UserProfile(
      name: name ?? this.name,
      email: email ?? this.email,
      avatarUrl: avatarUrl ?? this.avatarUrl,
      joinDate: joinDate ?? this.joinDate,
    );
  }
}
