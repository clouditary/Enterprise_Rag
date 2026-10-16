import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path so 'app' can be imported when running from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.ingestion import extract_text_from_pdf

# Resolve PDF path relative to project root
upload_dir = PROJECT_ROOT / "storage" / "uploads"
target_file = upload_dir / "(DAY 1)Project  — Enterprise RAG Knowledge Assistant .pdf"
if not target_file.exists():
    # Fallback to single space or any pdf in upload directory
    target_file = upload_dir / "(DAY 1)Project — Enterprise RAG Knowledge Assistant .pdf"
    if not target_file.exists():
        pdf_matches = list(upload_dir.glob("*.pdf"))
        if pdf_matches:
            target_file = pdf_matches[0]

pdf_path = str(target_file)

pages = extract_text_from_pdf(pdf_path)

print("total pages:",len(pages))

for page in pages:
    print (f"\n-- Page {page['page_number']}----")

    print (page["text"][:500])