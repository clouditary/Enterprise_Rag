# Enterprise RAG Knowledge Assistant 🚀

A production-grade Enterprise Retrieval-Augmented Generation (RAG) system built with **FastAPI**.

---

## 📁 Project Structure

The project maintains a strict separation between application code, documentation, and data storage:

```text
enterprise_rag/
│
├── app/                               # 💻 Application Backend Code
│   ├── __init__.py                    # Python package initializer
│   └── main.py                        # FastAPI application & route definitions
│
├── docs/                              # 📚 Documentation & Lecture Notes
│   ├── README.md                      # Documentation overview
│   └── lecture_01_fastapi_foundation.md  # Day 1: FastAPI Foundation guide
│
├── storage/                           # 💾 Local File & Document Storage
│   └── uploads/                       # Uploaded documents (PDFs, docs)
│       └── .gitkeep
│
├── .gitignore                         # Git ignore definitions
├── README.md                          # Main repository documentation
└── requirements.txt                   # Python dependencies
```

---

## 🛠️ Tech Stack & Prerequisites

- **Python**: 3.10+
- **FastAPI**: Modern, fast web framework for building APIs with Python
- **Uvicorn**: Lightning-fast ASGI server implementation
- **Pydantic**: Data validation and settings management

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/CLOUDITARY/enterprise_rag.git
cd enterprise_rag
```

### 2. Create and Activate Virtual Environment
**On Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Development Server
```bash
uvicorn app.main:app --reload --port 8000
```

The application will be running at `http://localhost:8000`.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :---: | :--- | :--- |
| `GET` | `/` | Root endpoint displaying service status |
| `GET` | `/health` | Health check endpoint for monitoring & orchestrators |
| `GET` | `/docs` | Interactive Swagger UI API documentation |
| `GET` | `/redoc` | Alternative ReDoc API documentation |

---

## 📖 Lectures & Documentation

All step-by-step lecture notes and architecture diagrams are maintained under [`docs/`](file:///c:/Users/Apoorva/Documents/enterprise_rag/docs):

- **Day 1**: [FastAPI Foundation & Project Setup](file:///c:/Users/Apoorva/Documents/enterprise_rag/docs/lecture_01_fastapi_foundation.md)
- *Upcoming*: Document Parsing, Chunking, Embeddings, Vector Search (Qdrant), Hybrid Retrieval, and LLM Generation.
