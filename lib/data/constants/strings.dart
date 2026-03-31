/// App strings - No placeholder text, all real content
class AppStrings {
  // App Info
  static const String appName = 'Emotion Controller';
  static const String appTagline = 'Understand and manage your emotions';

  // Home Screen
  static const String homeWelcome = 'Welcome';
  static const String homeGreeting = 'How are you feeling today?';
  static const String homeRecentEmotions = 'Recent Emotions';
  static const String homeQuickActions = 'Quick Actions';
  static const String homeStartChat = 'Start Chat';
  static const String homeUseCamera = 'Use Camera';
  static const String homeViewHistory = 'View History';
  static const String homeNoRecentEmotions =
      'No emotions detected yet. Start by chatting or using the camera!';

  // Chat Screen
  static const String chatTitle = 'AI Chat';
  static const String chatHint = 'Type how you\'re feeling...';
  static const String chatSend = 'Send';
  static const String chatTyping = 'AI is typing...';
  static const String chatEmpty =
      'Start a conversation to detect your emotions';
  static const String chatGreeting =
      'Hello! I\'m here to help you understand your emotions. Tell me how you\'re feeling today.';

  // Camera Screen
  static const String cameraTitle = 'Camera Detection';
  static const String cameraScanning = 'Scanning your face...';
  static const String cameraDetecting = 'Detecting emotions...';
  static const String cameraPositionFace = 'Position your face in the frame';
  static const String cameraReady = 'Ready to scan';
  static const String cameraResult = 'Emotion Detected';
  static const String cameraRetry = 'Try Again';
  static const String cameraPermissionDenied = 'Camera permission is required';
  static const String cameraError = 'Camera error occurred';

  // History Screen
  static const String historyTitle = 'Emotion History';
  static const String historyEmpty = 'No emotion history yet';
  static const String historyEmptySubtitle =
      'Your detected emotions will appear here';
  static const String historyFilterAll = 'All';
  static const String historyFilterChat = 'Chat';
  static const String historyFilterCamera = 'Camera';
  static const String historyToday = 'Today';
  static const String historyYesterday = 'Yesterday';
  static const String historyThisWeek = 'This Week';
  static const String historyEarlier = 'Earlier';

  // Profile Screen
  static const String profileTitle = 'Profile';
  static const String profileSettings = 'Settings';
  static const String profileAbout = 'About';
  static const String profileNotifications = 'Notifications';
  static const String profileLanguage = 'Language';
  static const String profileTheme = 'Theme';
  static const String profilePrivacy = 'Privacy';
  static const String profileHelp = 'Help & Support';
  static const String profileVersion = 'Version 0.1.0';
  static const String profileAppDescription =
      'Emotion Controller helps you understand and manage your emotions through AI-powered detection and personalized support.';

  // Emotion Detection
  static const String emotionDetected = 'Emotion Detected';
  static const String emotionIntensity = 'Intensity';
  static const String emotionMethod = 'Detection Method';
  static const String emotionMethodChat = 'AI Chat';
  static const String emotionMethodCamera = 'Camera';
  static const String emotionSuggestions = 'Personalized Suggestions';
  static const String emotionTimestamp = 'Detected at';

  // Emotion Names (7 backend emotions)
  static const String emotionAngry = 'Angry';
  static const String emotionDisgust = 'Disgust';
  static const String emotionFear = 'Fear';
  static const String emotionHappy = 'Happy';
  static const String emotionNeutral = 'Neutral';
  static const String emotionSad = 'Sad';
  static const String emotionSurprise = 'Surprise';

  // Common
  static const String loading = 'Loading...';
  static const String error = 'An error occurred';
  static const String retry = 'Retry';
  static const String cancel = 'Cancel';
  static const String done = 'Done';
  static const String save = 'Save';
  static const String delete = 'Delete';
  static const String edit = 'Edit';
  static const String close = 'Close';
  static const String ok = 'OK';
  static const String yes = 'Yes';
  static const String no = 'No';
}
