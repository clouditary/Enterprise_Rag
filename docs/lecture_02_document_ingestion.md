# (DAY 2) Project — Enterprise RAG Knowledge Assistant

## Step 2 — Document Ingestion & PDF Processing

---

### 1. Objective

In this step, we advance from our initial FastAPI foundation into the first functional stage of the **Enterprise RAG Pipeline**: **Document Ingestion and Text Extraction**.

Enterprise RAG systems do not begin with vectors or LLMs; they begin with unstructured corporate documents (PDFs, reports, policy manuals, product sheets). Before these documents can be indexed or searched, the system must:
1. Provide a secure, organized API for clients to upload documents with enterprise metadata (e.g., department, author).
2. Store raw documents persistently in a structured storage directory.
3. Validate file types to protect the ingestion pipeline from corrupted or unsupported formats.
4. Extract raw text page-by-page from PDFs while capturing critical metadata like page numbers for downstream source citations.
5. Provide a standalone testing workflow to verify extraction logic before chaining it into chunking and embedding pipelines.

---

### 2. What You Will Build

By the end of this step, your application includes:
- [x] A modular FastAPI router (`app/routers/documents.py`) using `APIRouter`.
- [x] A document upload endpoint (`POST /api/v1/documents/upload`) accepting file binaries and form metadata.
- [x] Validation to enforce `application/pdf` MIME types.
- [x] Persistent local disk storage in `storage/uploads/`.
- [x] A dedicated PDF extraction service (`app/core/ingestion.py`) using `pypdf`.
- [x] Structured page-level extraction (`page_number`, `text`) for accurate citation mapping.
- [x] A standalone test script (`app/test/test_extraction.py`) with environment-resilient path handling and Windows UTF-8 console support.

---

### 3. Project Directory Structure

Our project structure has expanded to cleanly separate API routing, core business logic, testing, and file storage:

```text
enterprise_rag/
│
├── app/
│   ├── __init__.py                 # Makes app a Python package
│   ├── main.py                     # FastAPI entry point (registers routers)
│   │
│   ├── core/                       # Core pipeline logic & services
│   │   ├── __init__.py             # Makes core a Python package
│   │   └── ingestion.py            # PDF text extraction logic (pypdf)
│   │
│   ├── routers/                    # Modular API route controllers
│   │   ├── __init__.py             # Makes routers a Python package
│   │   └── documents.py            # Document upload & management endpoints
│   │
│   └── test/                       # Standalone scripts & unit tests
│       └── test_extraction.py      # Standalone verification for PDF ingestion
│
├── docs/                           # Documentation & lecture notes
│   ├── README.md                   # Documentation index
│   ├── lecture_01_fastapi_foundation.md
│   └── lecture_02_document_ingestion.md
│
├── storage/                        # Persistent storage
│   └── uploads/                    # Directory where uploaded PDFs are stored
│       └── .gitkeep
│
├── .gitignore                      # Git ignore rules
├── README.md                       # Project overview
└── requirements.txt                # Updated dependencies (added pypdf, python-multipart)
```

#### New Files & Responsibilities

| File | Purpose |
| :--- | :--- |
| `app/routers/documents.py` | Contains the `APIRouter` for `/api/v1/documents` and handles file uploads. |
| `app/core/ingestion.py` | Standalone utility function `extract_text_from_pdf` that parses PDFs using `pypdf`. |
| `app/test/test_extraction.py` | Verification script that loads sample PDFs from storage and validates text extraction. |
| `storage/uploads/` | Dedicated filesystem folder where uploaded documents are saved. |
| `requirements.txt` | Includes `pypdf>=5.1.0` and `python-multipart>=0.0.9`. |

---

### 4. Modular Routing with `APIRouter`

In Step 1, all endpoints were placed inside `app/main.py`. In an enterprise codebase, keeping all routes in one file leads to a bloated, unmaintainable monolith.

FastAPI provides `APIRouter` to break down endpoints into modular, domain-specific controllers.

#### 4.1 Creating the Documents Router (`app/routers/documents.py`)

```python
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pathlib import Path

# Create router with prefix and OpenAPI tags
router = APIRouter(
    prefix="/api/v1/documents",
    tags=["Documents"]
)

# Define storage directory for uploads
UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
```

Key Concepts:
- **`prefix="/api/v1/documents"`**: All routes defined on this router will automatically have `/api/v1/documents` prepended (e.g., `POST /api/v1/documents/upload`).
- **`tags=["Documents"]`**: Groups all related document endpoints under a clean section in Swagger UI (`/docs`).
- **`UPLOAD_DIR.mkdir(parents=True, exist_ok=True)`**: Ensures the target directory exists on disk when the application starts, avoiding `FileNotFoundError` during uploads.

#### 4.2 Registering the Router in `app/main.py`

In `app/main.py`, we import and mount the router using `app.include_router()`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.documents import router as document_router

app = FastAPI(
    title="Enterprise RAG Knowledge Assistant",
    version="1.0.0",
    description="Enterprise RAG system built with FastAPI"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Document Router
app.include_router(document_router)

@app.get("/")
async def root():
    return {"message": "Enterprise RAG Assistant is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

---

### 5. Document Upload Endpoint (`POST /api/v1/documents/upload`)

The upload endpoint handles multipart form data, validates the content type, streams the file to disk, and tracks organizational metadata.

#### 5.1 Endpoint Implementation (`app/routers/documents.py`)

```python
@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    department: str = Form(...)
):
    # Validate MIME type
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed",
        )

    file_path = UPLOAD_DIR / file.filename

    # Read binary content asynchronously
    content = await file.read()

    # Save to storage/uploads
    with open(file_path, "wb") as f:
        f.write(content)

    return {
        "message": "file uploaded successfully",
        "filename": file.filename,
        "department": department
    }
```

#### 5.2 Key Technical Mechanisms

1. **`UploadFile = File(...)`**:
   - Unlike standard Python `bytes`, FastAPI's `UploadFile` uses a SpooledTemporaryFile. Small files stay in memory; large files overflow to disk.
   - Provides file metadata such as `filename` and `content_type`.
   - Requires `python-multipart` to parse multipart form bodies.

2. **`department: str = Form(...)`**:
   - Captures organizational metadata directly during upload.
   - In Enterprise RAG, metadata filtering (e.g. searching only HR policies or Engineering RFCs) is critical for access control and retrieval precision.

3. **MIME Type Validation**:
   - Checks `file.content_type != "application/pdf"`.
   - Rejects non-PDF files early with HTTP `400 Bad Request`.

4. **Asynchronous File Reading (`await file.read()`)**:
   - Reads the file asynchronously without blocking FastAPI's event loop.
   - Writes the raw bytes to `storage/uploads/<filename>` in binary write mode (`"wb"`).

---

### 6. PDF Text Extraction Engine (`app/core/ingestion.py`)

Once a PDF is stored, we need to extract its textual content. We implement `extract_text_from_pdf` using `pypdf`.

#### 6.1 Implementation

```python
from pathlib import Path
from pypdf import PdfReader

def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Extracts text from each page of a given PDF file.
    
    Args:
        file_path (str): Path to the target PDF file.
        
    Returns:
        list[dict]: A list of dictionaries containing page_number and text.
    """
    pdf_path = Path(file_path)

    # Verify file existence
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found : {file_path}")

    # Initialize PyPDF reader
    reader = PdfReader(str(pdf_path))

    extracted_pages = []

    # Iterate over pages with 1-based indexing
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        
        extracted_pages.append({
            "page_number": page_number,
            "text": text.strip()
        })

    return extracted_pages
```

#### 6.2 Why Page-by-Page Extraction is Essential for RAG

Many basic RAG tutorials dump the entire document into a single long string (`"".join(...)`). This is an anti-pattern for enterprise systems for two reasons:

1. **Grounded Source Citations**:
   When the LLM answers a user prompt (e.g. *"According to Page 4 of the HR Manual..."*), the citation must refer to the actual page number where the information lives. Preserving page numbers during initial extraction ensures all subsequent chunks inherit their exact page metadata.

2. **Chunk Attribution & Tracing**:
   If an answer references a hallucinated or incorrect statement, engineering teams can instantly trace the chunk back to the exact page of the original source PDF.

---

### 7. Standalone Testing & Real-World Troubleshooting (`app/test/test_extraction.py`)

Testing ingestion logic via the full web server every time slows down development. Building a standalone runner script allows rapid debugging of parsing logic.

#### 7.1 Test Implementation

```python
import sys
from pathlib import Path

# 1. Ensure UTF-8 output encoding on Windows consoles
if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 2. Add project root to sys.path for cross-directory imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.ingestion import extract_text_from_pdf

# 3. Resolve sample PDF path relative to project root
upload_dir = PROJECT_ROOT / "storage" / "uploads"
target_file = upload_dir / "(DAY 1)Project  — Enterprise RAG Knowledge Assistant .pdf"

if not target_file.exists():
    # Fallback to single-space variant or any PDF in storage
    target_file = upload_dir / "(DAY 1)Project — Enterprise RAG Knowledge Assistant .pdf"
    if not target_file.exists():
        pdf_matches = list(upload_dir.glob("*.pdf"))
        if pdf_matches:
            target_file = pdf_matches[0]

pdf_path = str(target_file)

# 4. Execute extraction
pages = extract_text_from_pdf(pdf_path)

print("total pages:", len(pages))

# 5. Display snippet of extracted text per page
for page in pages:
    print(f"\n-- Page {page['page_number']}----")
    print(page["text"][:500])
```

---

### 8. Key Engineering Challenges Encountered & Solved

During implementation, four critical issues were diagnosed and resolved:

#### Issue 1: `ModuleNotFoundError: No module named 'app'`
- **Cause**: When running `python test_extraction.py` from inside `app\test`, Python automatically sets `sys.path[0]` to `app\test`. It has no knowledge of the project root (`enterprise_rag`), so `from app.core...` fails.
- **Solution**: Compute `PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent` and insert it into `sys.path[0]` at runtime. Now the script runs smoothly from any folder.

#### Issue 2: `NameError: name 'extracted_page' is not defined`
- **Cause**: Typo in initialization: `extraced_page = []` vs `extracted_page.append(...)`.
- **Solution**: Standardized variable name to `extracted_page` (or `extracted_pages`).

#### Issue 3: `UnicodeEncodeError: 'charmap' codec can't encode characters`
- **Cause**: Windows command prompts default to legacy code pages (e.g. `cp1252`), which fail when printing Unicode characters commonly found in PDFs (em dashes `—`, bullet points `•`, smart quotes `“`).
- **Solution**: Explicitly reconfigure standard output to UTF-8 using `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`.

#### Issue 4: Fragile Relative Paths & File Spacing
- **Cause**: Using hardcoded relative paths like `"storage/uploads/..."` assumes the script is run from the workspace root. Additionally, PDF filenames with multiple spaces or unicode dashes can cause `FileNotFoundError`.
- **Solution**: Dynamically resolve paths relative to `PROJECT_ROOT` and include fallback globbing for matching `.pdf` files.

---

### 9. Verification & Testing

#### 9.1 Testing the Upload API via Swagger UI

1. Start the FastAPI development server:
   ```powershell
   uvicorn app.main:app --reload --port 8000
   ```
2. Open Swagger UI in your browser: `http://localhost:8000/docs`.
3. Locate the **Documents** section and expand `POST /api/v1/documents/upload`.
4. Click **Try it out**.
5. Choose a sample PDF file for `file` and type a department name (e.g., `Engineering`) in `department`.
6. Click **Execute**.
7. Confirm HTTP Status `200 OK` and response body:
   ```json
   {
     "message": "file uploaded successfully",
     "filename": "(DAY 1)Project  — Enterprise RAG Knowledge Assistant .pdf",
     "department": "Engineering"
   }
   ```
8. Verify that the file appears in your local `storage/uploads/` directory.

#### 9.2 Running the Standalone PDF Extraction Test

Run the test script directly from your terminal:

```powershell
# From the project root:
python -m app.test.test_extraction

# Or directly inside app/test:
cd app\test
python test_extraction.py
```

Expected output:
```text
total pages: 15

-- Page 1----
(DAY 1) Project — Enterprise RAG Knowledge Assistant
Step 1 — FastAPI Foundation
Objective...

-- Page 2----
Step 1 Completion Checklist...
```

---

### 10. Architecture & Data Flow

```mermaid
flowchart TD
    Client["Client (Browser / Postman / Swagger UI)"] -->|POST /api/v1/documents/upload\n(Multipart: File + Department)| MainApp["app/main.py (FastAPI App)"]
    MainApp --> Router["app/routers/documents.py (APIRouter)"]
    
    subgraph StorageSystem ["Local Storage"]
        Router -->|"Write binary stream"| UploadDir["storage/uploads/"]
    end
    
    subgraph ExtractionEngine ["Ingestion & Extraction Service"]
        TestScript["app/test/test_extraction.py"] -->|"Pass file path"| Ingestion["app/core/ingestion.py\n(extract_text_from_pdf)"]
        UploadDir -->|"Read PDF bytes"| Ingestion
        Ingestion -->|"pypdf.PdfReader"| PageLoop["Loop through reader.pages"]
        PageLoop -->|"Extract page text & number"| StructuredList["list[dict]:\n[{page_number: 1, text: '...'}, ...]"]
    end
```

---

### 11. Step 2 Completion Checklist

- [x] Installed `pypdf>=5.1.0` and `python-multipart>=0.0.9`.
- [x] Created `app/routers/documents.py` with `APIRouter`.
- [x] Mounted `document_router` in `app/main.py`.
- [x] Implemented `POST /api/v1/documents/upload` with PDF validation and disk saving.
- [x] Created `app/core/ingestion.py` with page-by-page text extraction.
- [x] Verified error handling for missing files and non-PDF uploads.
- [x] Implemented `app/test/test_extraction.py` with cross-platform path resolution and UTF-8 console output.
- [x] Successfully verified extraction of all pages from sample PDF document.

---

### 12. Preview: What Comes Next in Day 3

Now that raw text is extracted with associated page numbers, the next challenge is **Text Chunking**:
- Why raw pages cannot be sent directly to vector embeddings (embedding token limits & semantic drift).
- Fixed-size chunking vs **Recursive Character Chunking**.
- Managing `chunk_size` and `chunk_overlap` to maintain context across chunk boundaries.
- Propagating document-level and page-level metadata into each individual chunk.
