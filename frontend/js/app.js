/*
    Application Logic - Main Controller
    
    DESIGN DECISION: This is the application controller that coordinates UI and API
    REASON: Separation of concerns - this module contains business logic only
    
    WHY: By keeping UI, API, and app logic separate, we make the code:
    - Testable (can mock API and UI)
    - Maintainable (changes are isolated)
    - Auditable (clear flow of data)
*/

const App = {
    // Application state
    state: {
        currentQuery: null,
        currentAnswer: null,
        selectedFeedback: null,
        queryMetadata: {}
    },
    
    /**
     * Initialize application
     * 
     * WHY: Single entry point, sets up all event listeners
     */
    init() {
        // Initialize UI module
        UI.init();
        
        // Set up event listeners
        this._setupEventListeners();
        
        // Initialize with English by default
        UI.updateLanguage('en');
        
        console.log('NCERT Educational Assistant initialized');
    },
    
    /**
     * Set up all event listeners
     * 
     * WHY: Centralized event management for easier maintenance
     */
    _setupEventListeners() {
        /*
            DESIGN DECISION: Class selector triggers subject update
            REASON: Subjects vary by class, must sync immediately
        */
        UI.elements.classSelect.addEventListener('change', (e) => {
            UI.updateSubjectOptions(e.target.value);
            UI.updateQuestionInputState();
            this._updateQueryMetadata();
        });
        
        /*
            DESIGN DECISION: Subject selection enables question input
            REASON: All context must be set before asking question
        */
        UI.elements.subjectSelect.addEventListener('change', () => {
            UI.updateQuestionInputState();
            this._updateQueryMetadata();
        });
        
        /*
            DESIGN DECISION: Language selector updates entire UI
            REASON: Full multilingual support - UI must match selected language
        */
        UI.elements.languageSelect.addEventListener('change', (e) => {
            UI.updateLanguage(e.target.value);
            UI.updateQuestionInputState();
            this._updateQueryMetadata();
        });
        
        /*
            DESIGN DECISION: Real-time validation as user types
            REASON: Immediate feedback improves UX
        */
        UI.elements.questionInput.addEventListener('input', () => {
            UI.updateCharCounter();
            UI.updateSubmitButtonState();
        });
        
        /*
            DESIGN DECISION: Enter key submits (with Shift+Enter for new line)
            REASON: Common UX pattern, faster for users
        */
        UI.elements.questionInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!UI.elements.submitBtn.disabled) {
                    this.handleSubmit();
                }
            }
        });
        
        // Submit button
        UI.elements.submitBtn.addEventListener('click', () => {
            this.handleSubmit();
        });
        
        // Feedback buttons
        UI.elements.feedbackPositive.addEventListener('click', () => {
            this.handleFeedback(true);
        });
        
        UI.elements.feedbackNegative.addEventListener('click', () => {
            this.handleFeedback(false);
        });
        
        // Feedback submission
        UI.elements.feedbackSubmit.addEventListener('click', () => {
            this.submitFeedback();
        });
        
        // New question button
        UI.elements.newQuestionBtn.addEventListener('click', () => {
            this.handleNewQuestion();
        });
    },
    
    /**
     * Update query metadata from current selections
     * 
     * WHY: Store metadata for feedback logging
     */
    _updateQueryMetadata() {
        this.state.queryMetadata = {
            class: UI.elements.classSelect.value,
            subject: UI.elements.subjectSelect.value,
            language: UI.elements.languageSelect.value
        };
    },
    
    /**
     * Handle question submission
     * 
     * WHY: Core application flow - validate, call API, display result
     */
    async handleSubmit() {
        try {
            // Get current values
            const question = UI.elements.questionInput.value.trim();
            const classValue = UI.elements.classSelect.value;
            const subjectValue = UI.elements.subjectSelect.value;
            const languageValue = UI.elements.languageSelect.value;
            
            /*
                DESIGN DECISION: Final validation before API call
                REASON: Prevent invalid state even if UI validation fails
            */
            if (!question || !classValue || !subjectValue || !languageValue) {
                UI.displayError('Please fill all required fields before submitting.');
                return;
            }
            
            // Store current query
            this.state.currentQuery = {
                question,
                class: classValue,
                subject: subjectValue,
                language: languageValue
            };
            
            // Show loading state
            UI.showLoading();
            
            /*
                DESIGN DECISION: Try-catch for graceful error handling
                REASON: Network errors should not break UI
            */
            try {
                // Call API
                const response = await API.submitQuery(this.state.currentQuery);
                
                /*
                    DESIGN DECISION: Check response status
                    REASON: Backend can return success or rejection
                */
                if (response.status === 'success') {
                    this.state.currentAnswer = response;
                    UI.displayAnswer(response);
                } else if (response.status === 'rejected') {
                    this.state.currentAnswer = response;
                    UI.displayRejection(response);
                } else {
                    throw new Error('Invalid response from server');
                }
                
            } catch (error) {
                console.error('Query error:', error);
                UI.displayError(error.message || 'An error occurred. Please try again.');
            }
            
        } finally {
            // Always hide loading, even if error occurs
            UI.hideLoading();
        }
    },
    
    /**
     * Handle feedback button click
     * 
     * @param {boolean} isPositive - true for thumbs up, false for thumbs down
     * 
     * WHY: Collect evaluation data as specified
     */
    handleFeedback(isPositive) {
        this.state.selectedFeedback = isPositive;
        
        /*
            DESIGN DECISION: Update button visual state
            REASON: Clear feedback that action was registered
        */
        UI.elements.feedbackPositive.classList.toggle('active', isPositive);
        UI.elements.feedbackNegative.classList.toggle('active', !isPositive);
        
        /*
            DESIGN DECISION: Show text input for additional feedback
            REASON: Optional detail helps improve system
        */
        UI.showFeedbackInput();
    },
    
    /**
     * Submit feedback to backend
     * 
     * WHY: Log feedback for evaluation as specified
     */
    async submitFeedback() {
        if (this.state.selectedFeedback === null) {
            return;
        }
        
        try {
            const feedbackData = {
                question: this.state.currentQuery.question,
                answer: this.state.currentAnswer.answer || 'REJECTED',
                helpful: this.state.selectedFeedback,
                comment: UI.elements.feedbackText.value.trim(),
                metadata: {
                    ...this.state.queryMetadata,
                    status: this.state.currentAnswer.status,
                    rejection_type: this.state.currentAnswer.rejection_type || null,
                    grounding_score: this.state.currentAnswer.grounding_score || null
                }
            };
            
            /*
                DESIGN DECISION: Fire and forget feedback submission
                REASON: Don't block user if feedback fails
            */
            await API.submitFeedback(feedbackData);
            
            UI.showFeedbackConfirmation();
            
        } catch (error) {
            // Log but don't show error to user
            console.warn('Feedback submission failed:', error);
            // Still show confirmation for better UX
            UI.showFeedbackConfirmation();
        }
    },
    
    /**
     * Handle new question request
     * 
     * WHY: Reset state for next query
     */
    handleNewQuestion() {
        /*
            DESIGN DECISION: Keep context selections, clear only question
            REASON: Users often ask multiple questions about same class/subject
        */
        UI.resetForNewQuestion();
        
        // Reset state
        this.state.currentAnswer = null;
        this.state.selectedFeedback = null;
    }
};

/*
    DESIGN DECISION: Initialize on DOM ready
    REASON: Ensure all elements exist before accessing them
*/
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => App.init());
} else {
    App.init();
}

/*
    DESIGN DECISION: Expose App to global scope for debugging
    REASON: Easier to test in browser console during development
    
    WHY: In production, this can be removed or wrapped in IIFE
*/
window.App = App;
