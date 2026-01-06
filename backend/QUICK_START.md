# 🚀 Quick Start Guide

## For Mentors/Evaluators

This is a streamlined guide to quickly understand and test the NCERT RAG System.

---

## ⚡ 5-Minute Quick Start

### 1. Start the System (2 terminals needed)

**Terminal 1 - Backend:**
```powershell
cd "E:\WORK\intel unnati\backend"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```powershell
cd "E:\WORK\intel unnati\frontend"
python -m http.server 8080
```

### 2. Access the Application

Open browser: **http://localhost:8080**

### 3. Try Sample Questions

**Class 5 English:**
- "What is the importance of reading?"
- "Tell me about the story of Papa and his spectacles"

**Class 10 Hindi:**
- "जल संसाधन और जल संरक्षण के बारे में बताइए"

---

## 🧪 Testing the System

### Quick System Test
```bash
cd "E:\WORK\intel unnati\backend"
python test_system.py
```

**Expected Output:**
- ✓ 8-9 tests should pass
- Success rate: 80-90%
- All tests use `groq_llm` mode (if API available)

### Test LLM Integration
```bash
python test_llm.py
```

**Expected Output:**
- ✓ Groq API availability check
- ✓ Direct generation test
- Response time: 2-5 seconds

---

## 📚 Process New Textbooks

### Interactive Mode (Easiest)
```bash
python process_textbooks.py
```

Follow the prompts:
1. Enter class number (5, 10)
2. Select subject
3. Wait for processing (5-10 minutes per subject)

### Command Line Mode
```bash
# Example: Process Class 5 Mathematics
python process_textbooks.py --class 5 --subject Mathematics --language en
```

---

## 📊 What's Already Processed

| Class | Subject | Language | Chunks | Status |
|-------|---------|----------|--------|--------|
| 5 | English | English | 60 | ✅ Ready |
| 5 | Hindi | Hindi | 91 | ✅ Ready |
| 5 | PE | English | 39 | ✅ Ready |
| 10 | All Subjects | English | 623 | ✅ Ready |
| 10 | All Subjects | Hindi | 371 | ✅ Ready |

**Pending:**
- Class 5: Mathematics, Urdu
- Class 6: All subjects (~50 PDFs)
- Class 9: All subjects (~71 PDFs)

---

## 🔍 Key Files to Review

### Core Implementation
- `backend/main.py` - API server entry point
- `backend/app/services/pipeline_service.py` - Main RAG pipeline
- `backend/app/services/groq_service.py` - LLM integration

### Professional Scripts
- `process_textbooks.py` - Unified processing script
- `test_system.py` - Comprehensive testing
- `test_llm.py` - LLM-specific testing

### Frontend
- `frontend/index.html` - User interface
- `frontend/js/app.js` - Application logic

---

## 🏆 Key Features to Demonstrate

### 1. Multi-Class Support
- Each class has isolated vector storage
- No cross-contamination between classes
- Test: Ask Class 5 question, verify it doesn't use Class 10 data

### 2. Multi-Language Support
- English, Hindi, Urdu supported
- Language-specific OCR and processing
- Test: Switch between English/Hindi questions

### 3. LLM with Fallback
- Primary: Groq LLM (llama-3.3-70b)
- Fallback: Extractive mode (direct text)
- Test: Check response `metadata.mode` field

### 4. Semantic Search
- FAISS vector search
- Relevance threshold: 0.45
- Test: Ask off-topic question, see rejection

### 5. Subject Validation
- Class-specific filename prefixes
- Validates chunks match selected subject
- Test: System rejects mismatched questions

---

## 💡 Common Questions

### Q: Why some tests fail?
**A:** Class 5 PE has a known issue with subject validation (0.0 score). This is documented and being investigated.

### Q: Which mode is better - LLM or extractive?
**A:** LLM provides natural language answers. Extractive provides direct textbook quotes. Both are valid.

### Q: How long to process a textbook?
**A:** 5-15 minutes depending on PDF count and size. OCR is the slowest step.

### Q: Can I add more classes?
**A:** Yes! Just add PDFs to `data/` folder and run `process_textbooks.py`

---

## 🐛 Troubleshooting

### Backend won't start
```powershell
# Check port
netstat -ano | findstr :8000

# Kill if needed
taskkill /PID <PID> /F
```

### Frontend can't connect
- Ensure backend is on port 8000
- Check CORS settings in `app/core/config.py`
- Verify frontend is on port 8080

### Tests fail with connection error
- Start backend first: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`
- Wait 5 seconds for initialization
- Run tests

---

## 📞 Support

Check `README.md` for comprehensive documentation.

---

**System Status:** ✅ Production Ready  
**Last Updated:** January 6, 2026  
**Version:** 1.0.0
