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

# 1. Resolve sample PDF path
upload_dir = PROJECT_ROOT / "storage" / "uploads"
pdf_matches = list(upload_dir.glob("*.pdf"))
if not pdf_matches:
    print("❌ No sample PDF files found in storage/uploads/")
    sys.exit(1)

pdf_path = str(pdf_matches[0])
print(f"📄 Testing with PDF: {pdf_matches[0].name}")

# 2. Extract and chunk text
pages = extract_text_from_pdf(pdf_path)
print(f"✅ Total extracted pages: {len(pages)}")

chunks = chunk_pdf_pages(pages, chunk_size=600, chunk_overlap=120)
print(f"✅ Total generated chunks: {len(chunks)}")

# 3. Import and test Vector Store & Embedding Model
try:
    from app.core.vector_store import (
        embedding_model,
        create_collection,
        store_chunks,
        client,
        COLLECTION_NAME,
        VECTOR_SIZE,
    )

    sample_text = chunks[0]["text"]
    sample_vector = embedding_model.encode([sample_text])[0]
    print(f"✅ Embedding model loaded: sentence-transformers/all-MiniLM-L6-v2")
    print(f"✅ Generated embedding dimensions: {len(sample_vector)} (Expected: {VECTOR_SIZE})")

    # 4. Test Qdrant Vector Store connection
    print(f"\n🔍 Testing connection to Qdrant...")
    try:
        client.get_collections()
        print("✅ Connected to Qdrant server successfully!")

        create_collection()
        print(f"✅ Collection '{COLLECTION_NAME}' is ready.")

        department = "Engineering"
        total_stored = store_chunks(chunks[:5], department=department)
        print(f"✅ Successfully stored {total_stored} chunks for department '{department}' in Qdrant!")

    except Exception as exc:
        print(f"⚠️  Note: Could not connect to Qdrant server ({exc}).")
        print("👉 To run Qdrant locally using Docker, execute:")
        print("    docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/storage/qdrant_storage:/qdrant/storage:z qdrant/qdrant")

except OSError as err:
    if "c10.dll" in str(err) or "DLL" in str(err):
        print(f"\n⚠️  Windows PyTorch DLL Notice: {err}")
        print("👉 PyTorch requires Microsoft Visual C++ Redistributable on Windows.")
        print("👉 Download and install from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    else:
        raise
