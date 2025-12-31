/*
    API Integration Module
    
    DESIGN DECISION: Separate API logic from UI logic
    REASON: Testable, maintainable, easy to mock for development
    
    WHY: This module handles all backend communication and should never directly
    manipulate the DOM. It returns data or throws errors.
*/

const API = {
    /**
     * Submit a query to the backend RAG system
     * 
     * @param {Object} queryData - Query data
     * @param {string} queryData.question - User's question
     * @param {string} queryData.class - Selected class (5-10)
     * @param {string} queryData.subject - Selected subject
     * @param {string} queryData.language - Selected language (en/hi)
     * @returns {Promise<Object>} Response with answer or rejection
     * @throws {Error} If API call fails
     */
    async submitQuery(queryData) {
        /*
            DESIGN DECISION: Validate input before API call
            REASON: Fail fast, reduce unnecessary backend load
        */
        this._validateQueryData(queryData);
        
        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}${CONFIG.API_ENDPOINTS.QUERY}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    question: queryData.question.trim(),
                    class: parseInt(queryData.class),
                    subject: queryData.subject,
                    language: queryData.language,
                    timestamp: new Date().toISOString()
                }),
                /*
                    DESIGN DECISION: Explicit timeout
                    REASON: Prevents indefinite hanging, better UX
                */
                signal: AbortSignal.timeout(CONFIG.TIMEOUTS.API_TIMEOUT)
            });
            
            /*
                DESIGN DECISION: Check response status before parsing
                REASON: Handle HTTP errors gracefully
            */
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.message || `API error: ${response.status}`);
            }
            
            const data = await response.json();
            
            /*
                DESIGN DECISION: Validate response structure
                REASON: Backend may change, frontend should be defensive
            */
            this._validateResponseData(data);
            
            return data;
            
        } catch (error) {
            /*
                DESIGN DECISION: Categorize errors for better UX
                REASON: Different errors need different user messages
            */
            if (error.name === 'AbortError' || error.name === 'TimeoutError') {
                throw new Error('Request timed out. Please try again.');
            }
            
            if (error.message.includes('fetch')) {
                throw new Error('Unable to connect to server. Please check your connection.');
            }
            
            throw error;
        }
    },
    
    /**
     * Submit user feedback
     * 
     * @param {Object} feedbackData - Feedback data
     * @param {string} feedbackData.question - Original question
     * @param {string} feedbackData.answer - System's answer
     * @param {boolean} feedbackData.helpful - Was it helpful?
     * @param {string} feedbackData.comment - Optional comment
     * @param {Object} feedbackData.metadata - Query metadata
     * @returns {Promise<Object>} Confirmation
     * @throws {Error} If API call fails
     */
    async submitFeedback(feedbackData) {
        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}${CONFIG.API_ENDPOINTS.FEEDBACK}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    question: feedbackData.question,
                    answer: feedbackData.answer,
                    helpful: feedbackData.helpful,
                    comment: feedbackData.comment || '',
                    metadata: {
                        ...feedbackData.metadata,
                        timestamp: new Date().toISOString()
                    }
                }),
                signal: AbortSignal.timeout(CONFIG.TIMEOUTS.API_TIMEOUT)
            });
            
            if (!response.ok) {
                // Feedback submission failure should not break UX
                console.warn('Feedback submission failed:', response.status);
                return { success: false };
            }
            
            return await response.json();
            
        } catch (error) {
            // Log but don't throw - feedback is not critical
            console.warn('Feedback submission error:', error);
            return { success: false };
        }
    },
    
    /**
     * Validate query data before sending to backend
     * 
     * WHY: Catch errors early, provide clear messages
     */
    _validateQueryData(data) {
        if (!data.question || typeof data.question !== 'string') {
            throw new Error('Question is required');
        }
        
        if (data.question.trim().length < CONFIG.VALIDATION.MIN_QUESTION_LENGTH) {
            throw new Error(`Question must be at least ${CONFIG.VALIDATION.MIN_QUESTION_LENGTH} characters`);
        }
        
        if (data.question.length > CONFIG.VALIDATION.MAX_QUESTION_LENGTH) {
            throw new Error(`Question must not exceed ${CONFIG.VALIDATION.MAX_QUESTION_LENGTH} characters`);
        }
        
        if (!data.class || !['5', '6', '7', '8', '9', '10'].includes(data.class)) {
            throw new Error('Valid class selection is required');
        }
        
        if (!data.subject || typeof data.subject !== 'string') {
            throw new Error('Subject selection is required');
        }
        
        if (!data.language || !['en', 'hi'].includes(data.language)) {
            throw new Error('Language selection is required');
        }
    },
    
    /**
     * Validate response data from backend
     * 
     * WHY: Ensure backend contract is maintained
     */
    _validateResponseData(data) {
        if (!data || typeof data !== 'object') {
            throw new Error('Invalid response format');
        }
        
        /*
            DESIGN DECISION: Check for either success or rejection state
            REASON: Backend must provide one or the other, never both or neither
        */
        if (data.status === 'success') {
            if (!data.answer || typeof data.answer !== 'string') {
                throw new Error('Invalid answer format');
            }
            
            if (!Array.isArray(data.citations)) {
                throw new Error('Invalid citations format');
            }
            
        } else if (data.status === 'rejected') {
            if (!data.reason || typeof data.reason !== 'string') {
                throw new Error('Invalid rejection reason format');
            }
            
        } else {
            throw new Error('Invalid response status');
        }
    }
};

/*
    DESIGN DECISION: Expose API as singleton
    REASON: Single point of backend communication, easy to mock
*/
Object.freeze(API);
