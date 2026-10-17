import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path so 'app' can be imported when running from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.ingestion import extract_text_from_pdf, chunk_pdf_pages

# Resolve PDF path relative to project root
upload_dir = PROJECT_ROOT / "storage" / "uploads"
target_file = upload_dir / "(DAY 1)Project  — Enterprise RAG Knowledge Assistant .pdf"
if not target_file.exists():
    target_file = upload_dir / "(DAY 1)Project — Enterprise RAG Knowledge Assistant .pdf"
    if not target_file.exists():
        pdf_matches = list(upload_dir.glob("*.pdf"))
        if pdf_matches:
            target_file = pdf_matches[0]

pdf_path = str(target_file)

# 1. Extract pages
pages = extract_text_from_pdf(pdf_path)
print(f"Total extracted pages: {len(pages)}")

# 2. Chunk pages
chunks = chunk_pdf_pages(pages, chunk_size=600, chunk_overlap=120)
print(f"Total generated chunks: {len(chunks)}\n")

# 3. Print first 3 chunks as examples
for chunk in chunks[:3]:
    print(f"--- Chunk ID: {chunk['chunk_id']} | Page: {chunk['page_number']} | Index: {chunk['chunk_index']} ---")
    print(f"Text ({len(chunk['text'])} chars):\n{chunk['text'][:300]}...\n")
