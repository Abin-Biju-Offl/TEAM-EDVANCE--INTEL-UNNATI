# NCERT Educational RAG System
## Intel Unnati Project - Multi-Class Textbook Question Answering System

---

## 📋 Project Overview

A production-ready Retrieval-Augmented Generation (RAG) system for answering questions from NCERT textbooks across multiple classes (5, 6, 9, 10) with support for English, Hindi, and Urdu languages.

**Key Features:**
- ✅ Multi-class support with isolated vector storage
- ✅ Multi-language support (English, Hindi, Urdu)
- ✅ OCR-based text extraction from PDFs
- ✅ Groq LLM integration with extractive fallback
- ✅ FAISS-based semantic search
- ✅ Subject validation and confidence scoring
- ✅ RESTful API with CORS support

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (Port 8080)                   │
│                    HTML + CSS + JavaScript                  │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP/REST API
┌──────────────────────────▼──────────────────────────────────┐
│                  Backend FastAPI (Port 8000)                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Pipeline Service (Core)                 │   │
│  └──────────────────────────────────────────────────────┘   │
│           │          │          │          │                │
│  ┌────────▼───┐ ┌────▼─────┐ ┌─▼────────┐ ┌──▼──────────┐  │
│  │   OCR      │ │ Chunking │ │Embedding │ │   Groq LLM  │  │
│  │  Service   │ │ Service  │ │ Service  │ │   Service   │  │
│  └────────────┘ └──────────┘ └──────────┘ └─────────────┘  │
│           │                         │             │         │
│  ┌────────▼─────────────────────────▼─────────────▼──────┐  │
│  │              FAISS Vector Store                       │  │
│  │  class-5/     class-6/    class-9/    class-10/      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
backend/
├── main.py                      # FastAPI application entry point
├── requirements.txt             # Python dependencies
├── process_textbooks.py         # Combined processing script
├── test_system.py              # Comprehensive system testing
├── test_llm.py                 # LLM-specific testing
├── verify_chunks.py            # Chunk verification utility
│
├── app/                        # Application core
│   ├── api/
│   │   └── routes.py          # API endpoints
│   ├── core/
│   │   └── config.py          # Configuration
│   ├── models/
│   │   └── schemas.py         # Pydantic models
│   └── services/
│       ├── ocr_service.py     # PDF OCR processing
│       ├── chunking_service.py # Text chunking
│       ├── embedding_service.py # Sentence transformers
│       ├── faiss_service.py   # Vector search
│       ├── groq_service.py    # LLM integration
│       ├── pipeline_service.py # Main pipeline
│       └── safety_service.py  # Validation & safety
│
├── processed_data/             # OCR and cleaned data
│   └── class-{num}/
│       └── {subject}/
│           ├── ocr_results.json
│           ├── cleaned_documents.json
│           └── chunks.json
│
├── vector_store/               # FAISS indices
│   ├── class-5/
│   │   ├── english-en/
│   │   ├── hindi-hi/
│   │   └── physical-education-en/
│   └── class-10/
│       ├── all-subjects-english/
│       └── all-subjects-hindi/
│
└── logs/                       # Application logs

frontend/
├── index.html                  # Main UI
├── css/
│   └── styles.css
└── js/
    ├── app.js                  # Main application
    ├── api.js                  # API client
    ├── ui.js                   # UI components
    └── config.js               # Frontend config

data/
├── CLASS-V/                    # Class 5 PDFs
├── CLASS-VI/                   # Class 6 PDFs
├── CLASS-IX/                   # Class 9 PDFs
└── CLASS-X/                    # Class 10 PDFs
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Navigate to backend
cd "E:\WORK\intel unnati\backend"

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. Start Backend Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Start Frontend Server

```bash
cd "E:\WORK\intel unnati\frontend"
python -m http.server 8080
```

### 4. Access Application

- **Frontend**: http://localhost:8080
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## 📚 Processing Textbooks

### Interactive Mode (Recommended)

```bash
python process_textbooks.py
```

This will launch an interactive menu to select:
1. Class number
2. Subject
3. Language

### Command Line Mode

```bash
# Process specific subject
python process_textbooks.py --class 5 --subject English --language en

# Examples
python process_textbooks.py --class 5 --subject Hindi --language hi
python process_textbooks.py --class 10 --subject Science --language en
```

### Processing Pipeline

Each textbook goes through 4 stages:

1. **OCR**: Extract text from PDFs using Tesseract
2. **Cleaning**: Remove noise and normalize text
3. **Chunking**: Split into semantic chunks (400-600 tokens)
4. **Embedding**: Create vectors and FAISS index

**Outputs:**
- `processed_data/class-{num}/{subject}/ocr_results.json`
- `processed_data/class-{num}/{subject}/cleaned_documents.json`
- `processed_data/class-{num}/{subject}/chunks.json`
- `vector_store/class-{num}/{subject}-{lang}/faiss_index.index`

---

## 🧪 Testing

### Comprehensive System Test

Tests all classes, subjects, and languages:

```bash
python test_system.py
```

**Options:**
```bash
# Test specific class
python test_system.py --class 5

# Test with minimal output
python test_system.py --quiet

# Custom API endpoint
python test_system.py --api http://localhost:8000
```

**Test Coverage:**
- Class 5: English, Hindi, Physical Education
- Class 10: English (all subjects), Hindi (all subjects)
- Total: 9 test cases

### LLM Specific Test

Tests Groq API integration:

```bash
python test_llm.py
```

**Options:**
```bash
# Test direct generation only
python test_llm.py --direct

# Test rate limiting
python test_llm.py --rate-limit

# Run all tests
python test_llm.py --all
```

**Test Coverage:**
- API availability check
- Direct generation test
- Rate limiting behavior
- Token usage tracking

---

## 📊 Current Status

### Processed Classes

| Class | Subject | Language | Status | Chunks | Score Range |
|-------|---------|----------|--------|--------|-------------|
| 5 | English | English | ✅ Complete | 60 | 0.41-0.51 |
| 5 | Hindi | Hindi | ✅ Complete | 91 | 0.45-0.49 |
| 5 | Physical Education | English | ✅ Complete | 39 | N/A |
| 10 | All Subjects | English | ✅ Complete | 623 | 0.42-0.46 |
| 10 | All Subjects | Hindi | ✅ Complete | 371 | 0.46-0.51 |

### Pending Processing

- **Class 5**: Mathematics (16 PDFs), Urdu (20 PDFs)
- **Class 6**: ~50 PDFs across multiple subjects
- **Class 9**: ~71 PDFs across multiple subjects

### Known Issues

1. **Class 5 Physical Education**: Subject validation failing (returns 0.0 score)
   - Index exists with 39 chunks
   - Needs investigation of subject name mapping

2. **Groq Rate Limit**: 100k tokens/day on free tier
   - System automatically falls back to extractive mode
   - Upgrade to Dev tier for unlimited tokens

---

## 🔧 Technical Details

### Models & Libraries

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384D)
- **LLM**: Groq `llama-3.3-70b-versatile`
- **OCR**: Tesseract with PyMuPDF
- **Vector DB**: FAISS with IndexFlatIP
- **Framework**: FastAPI + Pydantic

### Configuration

Key settings in `app/core/config.py`:

```python
# API Configuration
API_URL = "http://localhost:8000"
CORS_ORIGINS = ["http://localhost:8080", "http://localhost:3000"]

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Search Parameters
RELEVANCE_THRESHOLD = 0.45
TOP_K = 30
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
```

### Class-Specific Validation

Each class has unique filename prefixes for validation:

```python
CLASS_5_PREFIXES = {
    "English": "eesa",
    "Hindi": "ehve",
    "Mathematics": "eemm",
    "Physical Education": "eeky",
    "Urdu": "eust"
}

CLASS_10_PREFIXES = {
    "Health and Physical Education": "jehp",
    "Mathematics": "jemh",
    "Science": "jesc",
    "Social Science": "jess"
}
```

### Special Handling

**Class 10 Structure:**
- Uses combined indices: `all-subjects-english` and `all-subjects-hindi`
- "English" and "Hindi" are treated as language selectors, not subjects
- Subject validation is bypassed for these language-based queries

---

## 📝 API Reference

### POST `/api/query`

Submit a question for answer generation.

**Request Body:**
```json
{
  "question": "What is the importance of reading?",
  "class": 5,
  "subject": "English",
  "language": "en"
}
```

**Response (Success):**
```json
{
  "status": "success",
  "answer": "Reading is important because...",
  "citations": [
    {
      "class": "5",
      "subject": "English",
      "chapter": "Chapter 1",
      "page": "1-16"
    }
  ],
  "grounding_score": 0.9,
  "metadata": {
    "retrieved_chunks": 30,
    "avg_retrieval_score": 0.4523,
    "mode": "groq_llm"
  }
}
```

**Response (Rejected):**
```json
{
  "status": "rejected",
  "reason": "Your question does not appear to be related to English.",
  "rejection_type": "off_topic",
  "metadata": {
    "top_score": 0.3245,
    "threshold": 0.45
  }
}
```

### GET `/health`

Check system health.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-01-06T12:00:00",
  "components": {
    "faiss_index": "ready",
    "llm": "groq_available"
  }
}
```

---

## 🛠️ Troubleshooting

### Backend Not Starting

```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Kill process if needed
taskkill /PID <PID> /F

# Restart backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Connection Issues

1. Check CORS configuration in `app/core/config.py`
2. Ensure backend is running on port 8000
3. Verify frontend is on port 8080

### Low Retrieval Scores

- Check if correct index is loaded for class/subject
- Verify embedding model is initialized
- Review question phrasing (more specific = better results)

### LLM Not Responding

1. Check Groq API key in `.env`
2. Verify rate limit status with `test_llm.py`
3. System will automatically use extractive fallback

---

## 👥 Team & Credits

**Intel Unnati Project Team**

**Technologies Used:**
- FastAPI
- Sentence Transformers
- FAISS
- Groq API
- Tesseract OCR
- PyMuPDF

**Data Source:**
- NCERT Textbooks (Classes 5, 6, 9, 10)

---

## 📄 License

Educational project for Intel Unnati program.

---

**Last Updated**: January 6, 2026  
**Version**: 1.0.0  
**Status**: Production Ready
