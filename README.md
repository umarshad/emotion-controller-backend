# Emotion Controller

A professional, production-ready Flutter frontend application for AI-powered emotion detection and management.

## Overview

Emotion Controller is a comprehensive Flutter application that provides real-time emotion detection through two primary methods:
- **AI Chatbot**: Text-based emotion detection through conversational interface
- **Live Camera**: Visual emotion detection with face scanning animations (UI only)

The app is built with a focus on clean architecture, scalability, and future backend integration readiness.

## Features

### Core Functionality
- 🤖 **AI Chat Interface**: Chat-based emotion detection with typing indicators and animated responses
- 📷 **Camera Detection**: Live camera preview with face scanning animations and emotion results
- 📊 **Emotion History**: Timeline view of all detected emotions with filtering options
- ⚙️ **Profile & Settings**: User profile management and app settings

### Emotion Support
The app supports 12 different emotions:
- 😠 Anger
- 😢 Sadness
- 😰 Stress
- 😟 Anxiety
- 😊 Happiness
- 😨 Fear
- 😕 Confusion
- 😔 Loneliness
- 😴 Tired
- 😌 Calm
- 💪 Motivation
- 😎 Relaxed

### Design Features
- **Material 3**: Modern Material Design 3 implementation
- **Fully Responsive**: Built with `flutter_screenutil` for perfect scaling across all devices
- **Smooth Animations**: Emotion-based animations, breathing effects, and micro-interactions
- **Professional UI**: No placeholder text, polished interface ready for production

## Tech Stack

- **Flutter**: Latest stable version
- **GetX**: State management and routing
- **flutter_screenutil**: Full responsiveness
- **persistent_bottom_nav_bar**: Bottom navigation with center-docked camera button
- **camera**: Camera preview UI
- **intl**: Date formatting and localization

## Project Structure

```
lib/
├── widgets/              # Reusable UI components
├── data/
│   ├── constants/        # Colors, strings, assets
│   ├── controllers/      # GetX controllers
│   ├── models/           # Data models
│   └── utils/            # Helpers, animations, mock data
└── view/                 # Screen widgets organized by feature
    ├── home/
    ├── chat/
    ├── camera/
    ├── history/
    └── profile/
```

## Getting Started

### Prerequisites
- Flutter SDK (latest stable)
- Dart SDK
- Android Studio / Xcode (for mobile development)

### Installation

1. Clone the repository
```bash
git clone https://github.com/umarshad/emotion-controller.git
cd emotion-controller
```

2. Install dependencies
```bash
flutter pub get
```

3. Run the app
```bash
flutter run
```

## Architecture

The app follows **Clean Architecture** principles with clear separation of concerns:

- **Controllers**: GetX controllers manage state and business logic
- **Models**: Data models for emotions and chat messages
- **Views**: UI screens organized by feature
- **Widgets**: Reusable UI components
- **Utils**: Helper functions and mock services

## State Management

The app uses **GetX** for:
- Reactive state management with `Obx` and `Rx` variables
- Navigation and routing
- Dependency injection
- Lifecycle management

## Responsive Design

All UI elements use `flutter_screenutil` for responsiveness:
- Font sizes: `.sp` extension
- Heights/widths: `.h` and `.w` extensions
- Padding/margins: `.r` extension

Design size: 375x812 (iPhone X)

## Mock Data

Currently, the app uses mock data services for emotion detection:
- **Chat**: Keyword-based emotion detection from user messages
- **Camera**: Simulated face detection with random emotion selection

These can be easily replaced with real API calls when backend is integrated.

## Future Integration

The app is designed to be backend-ready:
- Clear separation between UI and business logic
- Mock services easily replaceable with API calls
- Models support JSON serialization/deserialization
- Controllers ready for API integration

## License

This project is part of a Final Year Project (FYP) submission.

## Author

Built with ❤️ for emotion detection and mental health awareness.