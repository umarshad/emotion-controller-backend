import '../constants/video_urls.dart';

class Quotes {
  static const Map<String, List<String>> data = {
    'happy': [
      "Happiness is a butterfly, which when pursued, is always just beyond your grasp, but which, if you will sit down quietly, may alight upon you.",
      "The most important thing is to enjoy your life—to be happy—it's all that matters.",
      "Happiness is not something ready made. It comes from your own actions.",
      "Count your age by friends, not years. Count your life by smiles, not tears.",
      "Happiness is a direction, not a place.",
      "For every minute of happiness, you gain a day of life.",
      "Happiness is when what you think, what you say, and what you do are in harmony.",
      "Happiness depends upon ourselves. \u2014 Aristotle",
      "Do more of what makes you happy. \u2014 Anonymous",
      "For every minute you are angry, you lose sixty seconds of happiness. \u2014 Ralph Waldo Emerson",
    ],
    'sad': [
      "Every life has a measure of sorrow, and sometimes this is what awakens us.",
      "Tears are words that need to be written.",
      "It's okay to be sad. It means you're human.",
      "Sadness is but a wall between two gardens.",
      "The sun will shine again, even if it feels far away today.",
      "Your heart knows the way home, even in the dark.",
      "Healing comes in waves, and today it's okay to just let the tide come in.",
      "Sadness is just a bridge, not a destination. Every tear waters the seeds of tomorrow's joy.",
      "It's okay to feel sad. Emotions are visitors \u2014 let them come, let them go, but don't unpack and live there.",
      "Storms make trees take deeper roots. Let your sadness grow your resilience.",
      "If Allah is making you wait, then be prepared to receive more than what you asked for.",
    ],
    'angry': [
      "For every minute you remain angry, you give up sixty seconds of peace of mind.",
      "Anger is an acid that can do more harm to the vessel in which it is stored than to anything on which it is poured.",
      "The greatest remedy for anger is delay.",
      "Islamic Quote: 'The strong is not the one who overcomes the people by his strength, but the strong is the one who controls himself while in anger.' - Prophet Muhammad (PBUH)",
      "Islamic Quote: 'If one of you becomes angry while standing, he should sit down. If the anger leaves him, well and good; otherwise he should lie down.'",
      "Islamic Quote: 'Show forgiveness, enjoin what is good, and turn away from the foolish.' (Quran 7:199)",
      "Anger is like a storm; it passes, but the landscape is changed. Choose to build rather than break.",
      "\u0644\u064e\u064a\u0652\u0633\u064e \u0627\u0644\u0634\u064e\u062f\u064a\u062f\u064f \u0628\u0627\u0644\u0635\u0651\u0650\u0631\u064e\u0639\u064e\u0629 \u2014 The person who is strong is the one who controls himself when he is angry. \u2014 Prophet Muhammad (PBUH)",
      "\u0627\u0644\u0644\u0651\u064e\u0647\u064f\u0645\u0651\u064e \u0623\u064e\u0630\u0652\u0647\u064e\u0628\u0652 \u063a\u064e\u064a\u0652\u0638\u064e \u0642\u064e\u0644\u0652\u0628\u064a \u2014 Oh Allah, remove anger from my heart.",
    ],
    'fear': [
      "The only thing we have to fear is fear itself.",
      "Fear is a reaction. Courage is a decision.",
      "Do the thing you fear and the death of fear is certain.",
      "Fear has two meanings: 'Forget Everything And Run' or 'Face Everything And Rise.' The choice is yours.",
      "He who is not everyday conquering some fear has not learned the secret of life.",
      "Courage is not the absence of fear, but the triumph over it.",
      "Everything you've ever wanted is on the other side of fear.",
    ],
    'surprise': [
      "Expect nothing. Live frugally on surprise.",
      "The moments of surprise are the most memorable.",
      "Life is full of surprises, some good, some bad.",
      "Surprise is the greatest gift which life can grant us.",
      "May you live all the days of your life in wonder.",
      "The secret to a rich life is to have more beginnings than endings.",
      "Surprise is the spark that keeps life interesting.",
    ],
    'neutral': [
      "Calmness is the cradle of power.",
      "Peace comes from within. Do not seek it without.",
      "Maintain your inner peace.",
      "Balance is key to a healthy life.",
      "Serenity is not freedom from the storm, but peace amid the storm.",
      "In the middle of movement and chaos, keep stillness inside of you.",
      "Quiet the mind, and the soul will speak.",
      "The best way to predict the future is to create it. \u2014 Abraham Lincoln",
      "Believe you can and you're halfway there. \u2014 Theodore Roosevelt",
      "Success is not final, failure is not fatal: It is the courage to continue that counts. \u2014 Winston Churchill",
      "It always seems impossible until it's done. \u2014 Nelson Mandela",
    ],
    'disgust': [
      "Standards are not about rejection, but about the refinement of our own path.",
      "Disgust is the soul's way of reminding us that we were meant for higher things.",
      "Refining one's environment is the first step toward refining one's mind.",
      "Excellence is never an accident; it is the result of high intention and sincere effort.",
      "True refinement comes from knowing what to include and what to leave behind.",
      "A healthy boundary is the distance at which I can love you and me simultaneously.",
      "Seek the pure, the simple, and the sincere; they are the antidotes to the distorted.",
    ],
  };

  // Reflection videos are now sourced from EmotionVideoUrls
  // to guarantee in-app playback (all URLs are confirmed embeddable).
  static List<String> getQuotesFor(String emotion) {
    return data[emotion.toLowerCase()] ??
        [
          "Keep going. You're doing great.",
          "Embrace your emotions.",
          "Stay positive.",
        ];
  }

  static List<Map<String, String>> getVideosFor(String emotion) {
    // Delegate to EmotionVideoUrls so only confirmed-embeddable
    // URLs are returned — prevents "Playback Restricted" in-app.
    return EmotionVideoUrls.getReflectionVideos(emotion);
  }

  static List<String> get allQuotes {
    List<String> all = [];
    data.values.forEach(all.addAll);
    return all;
  }
}
