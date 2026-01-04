/*
    Configuration File
    
    DESIGN DECISION: Centralized configuration
    REASON: Easy to update API endpoints and settings without touching core logic
*/

const CONFIG = {
    // API Configuration
    API_BASE_URL: 'http://localhost:8000',  // Update with your backend URL
    API_ENDPOINTS: {
        QUERY: '/api/query',
        FEEDBACK: '/api/feedback'
    },
    
    // Subject mapping by class
    // WHY: NCERT subjects vary by class level
    SUBJECTS_BY_CLASS: {
        '5': ['Mathematics', 'English', 'Hindi', 'Environmental Studies'],
        '6': ['Mathematics', 'Science', 'English', 'Hindi', 'Social Science'],
        '7': ['Mathematics', 'Science', 'English', 'Hindi', 'Social Science'],
        '8': ['Mathematics', 'Science', 'English', 'Hindi', 'Social Science'],
        '9': ['Mathematics', 'Science', 'English', 'Hindi', 'Social Science'],
        '10': ['Mathematics', 'Science', 'English', 'Hindi', 'Social Science', 'Health and Physical Science']
    },
    
    // Multilingual UI text
    // WHY: Support both English and Hindi interfaces
    TRANSLATIONS: {
        'en': {
            systemTitle: 'NCERT Educational Assistant',
            systemSubtitle: 'Intel Unnati Industrial Training Project',
            trustBadge: 'NCERT Verified Content Only',
            safetyBanner: 'Answers are generated <strong>strictly from NCERT textbooks only</strong>. Questions outside the NCERT curriculum will be declined.',
            askQuestion: 'Ask Your Question',
            step1: 'Step 1: Select your textbook details (all fields are mandatory)',
            step2: 'Step 2: Ask your question',
            classLabel: 'Class',
            subjectLabel: 'Subject',
            languageLabel: 'Language',
            selectClass: 'Select Class',
            selectSubject: 'Select Subject',
            selectLanguage: 'Select Language',
            questionPlaceholder: "Ask a question strictly from your NCERT textbook (e.g., 'What is photosynthesis?' or 'Explain Pythagoras theorem')",
            getAnswer: 'Get Answer',
            processing: 'Processing your question...',
            searching: 'Searching NCERT textbooks',
            answerTitle: 'Answer from NCERT Textbook',
            unableToAnswer: 'Unable to Answer',
            dontKnow: "I don't know based on NCERT textbooks.",
            sourceCitations: 'Source Citations',
            answerReliability: 'Answer Reliability:',
            reliabilityHigh: 'Fully Verified',
            reliabilityMedium: 'Partially Verified',
            feedbackQuestion: 'Was this answer helpful?',
            feedbackYes: 'Yes',
            feedbackNo: 'No',
            feedbackMore: 'Tell us more (optional):',
            feedbackPlaceholder: 'Your feedback helps us improve...',
            submitFeedback: 'Submit Feedback',
            feedbackThank: '✓ Thank you for your feedback!',
            askAnother: 'Ask Another Question',
            rejectionGuidanceTitle: 'This can happen if:',
            rejectionGuidance: [
                'Your question is not covered in the selected NCERT textbook',
                'The topic is from a different class or subject',
                'The question requires information beyond NCERT curriculum'
            ],
            rejectionSuggestion: '<strong>Try:</strong> Rephrasing your question or checking if you\'ve selected the correct class and subject.',
            footerText: '<strong>Intel Unnati Industrial Training Project</strong> | Team EDVANCE | December 2025',
            footerDisclaimer: 'This system provides answers based solely on NCERT textbook content. It does not provide exam answers, paper leaks, or general knowledge outside the curriculum.'
        },
        'hi': {
            systemTitle: 'NCERT शैक्षिक सहायक',
            systemSubtitle: 'इंटेल उन्नति औद्योगिक प्रशिक्षण परियोजना',
            trustBadge: 'केवल NCERT सत्यापित सामग्री',
            safetyBanner: 'उत्तर <strong>केवल NCERT पाठ्यपुस्तकों से</strong> उत्पन्न किए जाते हैं। NCERT पाठ्यक्रम से बाहर के प्रश्नों को अस्वीकार कर दिया जाएगा।',
            askQuestion: 'अपना प्रश्न पूछें',
            step1: 'चरण 1: अपनी पाठ्यपुस्तक का विवरण चुनें (सभी फ़ील्ड अनिवार्य हैं)',
            step2: 'चरण 2: अपना प्रश्न पूछें',
            classLabel: 'कक्षा',
            subjectLabel: 'विषय',
            languageLabel: 'भाषा',
            selectClass: 'कक्षा चुनें',
            selectSubject: 'विषय चुनें',
            selectLanguage: 'भाषा चुनें',
            questionPlaceholder: "अपनी NCERT पाठ्यपुस्तक से सख्ती से एक प्रश्न पूछें (उदाहरण: 'प्रकाश संश्लेषण क्या है?' या 'पाइथागोरस प्रमेय समझाएं')",
            getAnswer: 'उत्तर प्राप्त करें',
            processing: 'आपके प्रश्न को संसाधित किया जा रहा है...',
            searching: 'NCERT पाठ्यपुस्तकों में खोज',
            answerTitle: 'NCERT पाठ्यपुस्तक से उत्तर',
            unableToAnswer: 'उत्तर देने में असमर्थ',
            dontKnow: 'NCERT पाठ्यपुस्तकों के आधार पर मुझे नहीं पता।',
            sourceCitations: 'स्रोत उद्धरण',
            answerReliability: 'उत्तर विश्वसनीयता:',
            reliabilityHigh: 'पूर्णतः सत्यापित',
            reliabilityMedium: 'आंशिक रूप से सत्यापित',
            feedbackQuestion: 'क्या यह उत्तर सहायक था?',
            feedbackYes: 'हाँ',
            feedbackNo: 'नहीं',
            feedbackMore: 'हमें और बताएं (वैकल्पिक):',
            feedbackPlaceholder: 'आपकी प्रतिक्रिया हमें बेहतर बनाने में मदद करती है...',
            submitFeedback: 'प्रतिक्रिया सबमिट करें',
            feedbackThank: '✓ आपकी प्रतिक्रिया के लिए धन्यवाद!',
            askAnother: 'एक और प्रश्न पूछें',
            rejectionGuidanceTitle: 'ऐसा तब हो सकता है जब:',
            rejectionGuidance: [
                'आपका प्रश्न चयनित NCERT पाठ्यपुस्तक में शामिल नहीं है',
                'विषय किसी अन्य कक्षा या विषय से है',
                'प्रश्न को NCERT पाठ्यक्रम से परे जानकारी की आवश्यकता है'
            ],
            rejectionSuggestion: '<strong>कोशिश करें:</strong> अपने प्रश्न को दोबारा लिखें या जांचें कि क्या आपने सही कक्षा और विषय चुना है।',
            footerText: '<strong>इंटेल उन्नति औद्योगिक प्रशिक्षण परियोजना</strong> | टीम EDVANCE | दिसंबर 2025',
            footerDisclaimer: 'यह प्रणाली केवल NCERT पाठ्यपुस्तक सामग्री के आधार पर उत्तर प्रदान करती है। यह परीक्षा के उत्तर, पेपर लीक, या पाठ्यक्रम के बाहर सामान्य ज्ञान प्रदान नहीं करती है।'
        }
    },
    
    // Validation rules
    VALIDATION: {
        MIN_QUESTION_LENGTH: 10,
        MAX_QUESTION_LENGTH: 500,
        MAX_FEEDBACK_LENGTH: 200
    },
    
    // Timeout settings
    TIMEOUTS: {
        API_TIMEOUT: 30000,  // 30 seconds
        DEBOUNCE_DELAY: 300  // 300ms for input debouncing
    },
    
    // Safety thresholds (must match backend)
    // WHY: Frontend should show same logic as backend for transparency
    SAFETY_THRESHOLDS: {
        MIN_RETRIEVAL_CONFIDENCE: 0.6,
        MIN_CHUNKS_REQUIRED: 1,
        MIN_GROUNDING_SCORE: 0.7
    }
};

// Freeze configuration to prevent accidental modifications
// WHY: Configuration should be immutable at runtime
Object.freeze(CONFIG);
Object.freeze(CONFIG.SUBJECTS_BY_CLASS);
Object.freeze(CONFIG.TRANSLATIONS);
Object.freeze(CONFIG.VALIDATION);
Object.freeze(CONFIG.TIMEOUTS);
Object.freeze(CONFIG.SAFETY_THRESHOLDS);
