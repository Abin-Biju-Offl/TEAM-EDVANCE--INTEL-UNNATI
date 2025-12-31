/*
    UI Module
    
    DESIGN DECISION: Centralized DOM manipulation
    REASON: All UI updates go through this module for consistency and testability
    
    WHY: Separating UI logic from application logic makes the code more maintainable
    and easier to audit. Each function has a single responsibility.
*/

const UI = {
    // Cache DOM elements for performance
    elements: {},
    
    /**
     * Initialize UI module by caching DOM elements
     * 
     * WHY: Query DOM once, reuse references for better performance
     */
    init() {
        this.elements = {
            // Selectors
            classSelect: document.getElementById('class-select'),
            subjectSelect: document.getElementById('subject-select'),
            languageSelect: document.getElementById('language-select'),
            
            // Question input
            questionInput: document.getElementById('question-input'),
            charCount: document.getElementById('char-count'),
            submitBtn: document.getElementById('submit-btn'),
            btnText: document.getElementById('btn-text'),
            
            // Answer section
            answerSection: document.getElementById('answer-section'),
            successState: document.getElementById('success-state'),
            rejectionState: document.getElementById('rejection-state'),
            
            // Answer content
            answerText: document.getElementById('answer-text'),
            citationsList: document.getElementById('citations-list'),
            groundingStatus: document.getElementById('grounding-status'),
            
            // Rejection content
            rejectionReason: document.getElementById('rejection-reason'),
            
            // Feedback
            feedbackPositive: document.getElementById('feedback-positive'),
            feedbackNegative: document.getElementById('feedback-negative'),
            feedbackTextContainer: document.getElementById('feedback-text-container'),
            feedbackText: document.getElementById('feedback-text'),
            feedbackSubmit: document.getElementById('feedback-submit'),
            feedbackConfirmation: document.getElementById('feedback-confirmation'),
            
            // Actions
            newQuestionBtn: document.getElementById('new-question-btn'),
            
            // Loading
            loadingOverlay: document.getElementById('loading-overlay')
        };
    },
    
    /**
     * Update subject dropdown based on selected class
     * 
     * WHY: NCERT subjects vary by class, must sync UI with data
     */
    updateSubjectOptions(classValue) {
        const subjects = CONFIG.SUBJECTS_BY_CLASS[classValue] || [];
        const currentLang = this.elements.languageSelect.value;
        const t = currentLang ? CONFIG.TRANSLATIONS[currentLang] : CONFIG.TRANSLATIONS.en;
        
        this.elements.subjectSelect.innerHTML = `
            <option value="" disabled selected>${t.selectSubject}</option>
            ${subjects.map(subject => `<option value="${subject}">${subject}</option>`).join('')}
        `;
        
        /*
            DESIGN DECISION: Enable subject selector only after class is selected
            REASON: Enforces mandatory field order, prevents invalid state
        */
        this.elements.subjectSelect.disabled = !classValue;
    },
    
    /**
     * Enable/disable question input based on context selection
     * 
     * WHY: Question should only be asked after all context is provided
     */
    updateQuestionInputState() {
        const classValue = this.elements.classSelect.value;
        const subjectValue = this.elements.subjectSelect.value;
        const languageValue = this.elements.languageSelect.value;
        
        /*
            DESIGN DECISION: All three fields must be selected before enabling input
            REASON: Prevents incomplete queries, improves answer accuracy
        */
        const allSelected = classValue && subjectValue && languageValue;
        
        this.elements.questionInput.disabled = !allSelected;
        
        if (allSelected) {
            this.elements.questionInput.focus();
        }
    },
    
    /**
     * Enable/disable submit button based on question validity
     * 
     * WHY: Prevent submission of invalid questions
     */
    updateSubmitButtonState() {
        const question = this.elements.questionInput.value.trim();
        const isValid = question.length >= CONFIG.VALIDATION.MIN_QUESTION_LENGTH;
        
        this.elements.submitBtn.disabled = !isValid;
    },
    
    /**
     * Update character counter
     * 
     * WHY: Help users stay within limits
     */
    updateCharCounter() {
        const count = this.elements.questionInput.value.length;
        this.elements.charCount.textContent = count;
        
        /*
            DESIGN DECISION: Visual feedback when approaching limit
            REASON: Prevents frustration from hitting hard limit
        */
        if (count > CONFIG.VALIDATION.MAX_QUESTION_LENGTH * 0.9) {
            this.elements.charCount.style.color = 'var(--warning-orange)';
        } else {
            this.elements.charCount.style.color = 'var(--gray-500)';
        }
    },
    
    /**
     * Show loading overlay
     * 
     * WHY: Clear feedback during API calls, prevents double submission
     */
    showLoading() {
        this.elements.loadingOverlay.classList.remove('hidden');
        this.elements.submitBtn.disabled = true;
        this.elements.submitBtn.classList.add('loading');
    },
    
    /**
     * Hide loading overlay
     */
    hideLoading() {
        this.elements.loadingOverlay.classList.add('hidden');
        this.elements.submitBtn.classList.remove('loading');
        this.updateSubmitButtonState();
    },
    
    /**
     * Display successful answer with citations
     * 
     * @param {Object} data - Response data from backend
     * 
     * WHY: This is the primary success state - must be clear and trustworthy
     */
    displayAnswer(data) {
        /*
            DESIGN DECISION: Show answer section, hide rejection state
            REASON: Only one state visible at a time for clarity
        */
        this.elements.answerSection.classList.remove('hidden');
        this.elements.successState.classList.remove('hidden');
        this.elements.rejectionState.classList.add('hidden');
        
        // Display answer text
        this.elements.answerText.innerHTML = this._formatAnswerText(data.answer);
        
        // Display citations
        this._displayCitations(data.citations);
        
        // Display grounding indicator
        this._displayGroundingStatus(data.grounding_score);
        
        // Reset feedback
        this._resetFeedback();
        
        // Scroll to answer
        this.elements.answerSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    },
    
    /**
     * Display rejection state ("I don't know")
     * 
     * @param {Object} data - Rejection data from backend
     * 
     * WHY: Rejection must be clear but professional, not an error state
     */
    displayRejection(data) {
        this.elements.answerSection.classList.remove('hidden');
        this.elements.successState.classList.add('hidden');
        this.elements.rejectionState.classList.remove('hidden');
        
        /*
            DESIGN DECISION: Show specific rejection reason
            REASON: Helps users understand why and how to fix
        */
        const reasonMessages = {
            'low_confidence': 'The retrieved information does not have sufficient confidence to provide an accurate answer.',
            'insufficient_context': 'Not enough relevant content was found in the NCERT textbooks for your question.',
            'off_topic': 'Your question appears to be outside the scope of NCERT textbooks.',
            'missing_citations': 'Unable to properly cite sources for a reliable answer.',
            'poor_grounding': 'The generated answer could not be adequately verified against the textbook content.'
        };
        
        const reason = reasonMessages[data.rejection_type] || data.reason || 
                      'Your question could not be answered from the available NCERT textbook content.';
        
        this.elements.rejectionReason.textContent = reason;
        
        // Reset feedback
        this._resetFeedback();
        
        // Scroll to answer section
        this.elements.answerSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    },
    
    /**
     * Display error message
     * 
     * @param {string} message - Error message
     * 
     * WHY: Errors should be clear but not alarming
     */
    displayError(message) {
        /*
            DESIGN DECISION: Use rejection state for errors
            REASON: Consistent UI, errors are just another form of "can't answer"
        */
        this.elements.answerSection.classList.remove('hidden');
        this.elements.successState.classList.add('hidden');
        this.elements.rejectionState.classList.remove('hidden');
        
        this.elements.rejectionReason.textContent = message;
        
        this.elements.answerSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    },
    
    /**
     * Reset UI to initial state for new question
     * 
     * WHY: Clean slate for each query
     */
    resetForNewQuestion() {
        this.elements.answerSection.classList.add('hidden');
        this.elements.questionInput.value = '';
        this.updateCharCounter();
        this.updateSubmitButtonState();
        this.elements.questionInput.focus();
        
        // Scroll to question input
        this.elements.questionInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
    },
    
    /**
     * Format answer text with proper paragraphs
     * 
     * WHY: Better readability, professional appearance
     */
    _formatAnswerText(text) {
        return text
            .split('\n\n')
            .map(para => `<p>${this._escapeHtml(para)}</p>`)
            .join('');
    },
    
    /**
     * Display citations in standardized format
     * 
     * @param {Array} citations - Citation objects from backend
     * 
     * WHY: Citations are CRITICAL for transparency and trust
     */
    _displayCitations(citations) {
        if (!citations || citations.length === 0) {
            /*
                DESIGN DECISION: No citations = error state
                REASON: System should never show answer without citations
            */
            this.elements.citationsList.innerHTML = `
                <div class="citation-item">
                    <span class="citation-text">
                        <strong>Warning:</strong> No citations available for this answer.
                    </span>
                </div>
            `;
            return;
        }
        
        /*
            DESIGN DECISION: Numbered list with full citation details
            REASON: Easy to reference, complete source information
        */
        this.elements.citationsList.innerHTML = citations.map((citation, index) => `
            <div class="citation-item">
                <span class="citation-number">${index + 1}</span>
                <span class="citation-text">
                    <strong>NCERT Class ${this._escapeHtml(citation.class)}</strong>, 
                    ${this._escapeHtml(citation.subject)}, 
                    Chapter ${this._escapeHtml(citation.chapter)}, 
                    Page ${this._escapeHtml(citation.page)}
                    ${citation.section ? ` (${this._escapeHtml(citation.section)})` : ''}
                </span>
            </div>
        `).join('');
    },
    
    /**
     * Display grounding status in plain language
     * 
     * @param {number} score - Grounding score (0-1)
     * 
     * WHY: Build trust with transparency about answer reliability
     */
    _displayGroundingStatus(score) {
        let statusText, statusClass;
        
        /*
            DESIGN DECISION: Plain language, not numbers
            REASON: Reviewers understand reliability better than scores
        */
        if (score >= 0.9) {
            statusText = 'Fully Verified';
            statusClass = 'high';
        } else if (score >= 0.7) {
            statusText = 'Verified';
            statusClass = 'verified';
        } else {
            statusText = 'Partially Verified';
            statusClass = 'medium';
        }
        
        this.elements.groundingStatus.textContent = statusText;
        this.elements.groundingStatus.className = `grounding-status ${statusClass}`;
    },
    
    /**
     * Show feedback text input
     * 
     * WHY: Progressive disclosure - only show when needed
     */
    showFeedbackInput() {
        this.elements.feedbackTextContainer.classList.remove('hidden');
        this.elements.feedbackText.focus();
    },
    
    /**
     * Show feedback confirmation
     * 
     * WHY: Acknowledge user action
     */
    showFeedbackConfirmation() {
        this.elements.feedbackTextContainer.classList.add('hidden');
        this.elements.feedbackConfirmation.classList.remove('hidden');
        
        // Hide confirmation after 3 seconds
        setTimeout(() => {
            this.elements.feedbackConfirmation.classList.add('hidden');
        }, 3000);
    },
    
    /**
     * Reset feedback UI
     */
    _resetFeedback() {
        this.elements.feedbackPositive.classList.remove('active');
        this.elements.feedbackNegative.classList.remove('active');
        this.elements.feedbackTextContainer.classList.add('hidden');
        this.elements.feedbackConfirmation.classList.add('hidden');
        this.elements.feedbackText.value = '';
    },
    
    /**
     * Escape HTML to prevent XSS
     * 
     * WHY: Security - never trust user input or backend data
     */
    _escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },
    
    /**
     * Update UI language
     * 
     * @param {string} lang - Language code ('en' or 'hi')
     * 
     * WHY: Full multilingual support as specified
     */
    updateLanguage(lang) {
        const t = CONFIG.TRANSLATIONS[lang];
        
        // Update static text elements
        document.querySelector('.system-title').textContent = t.systemTitle;
        document.querySelector('.system-subtitle').textContent = t.systemSubtitle;
        document.querySelector('.badge-text').textContent = t.trustBadge;
        document.querySelector('.safety-text').innerHTML = t.safetyBanner;
        document.querySelector('.section-title').textContent = t.askQuestion;
        
        // Update labels
        document.querySelector('.context-instruction').innerHTML = `<strong>${t.step1.split(':')[0]}:</strong>${t.step1.split(':')[1]}`;
        document.querySelectorAll('.context-instruction')[1].innerHTML = `<strong>${t.step2.split(':')[0]}:</strong>${t.step2.split(':')[1]}`;
        
        // Update selectors
        this.elements.classSelect.options[0].text = t.selectClass;
        document.querySelector('label[for="class-select"] .label-text').textContent = t.classLabel;
        document.querySelector('label[for="subject-select"] .label-text').textContent = t.subjectLabel;
        document.querySelector('label[for="language-select"] .label-text').textContent = t.languageLabel;
        
        // Update placeholders
        this.elements.questionInput.placeholder = t.questionPlaceholder;
        document.getElementById('btn-text').textContent = t.getAnswer;
        
        // Update footer
        document.querySelector('.footer-text').innerHTML = t.footerText;
        document.querySelector('.footer-disclaimer').textContent = t.footerDisclaimer;
    }
};
