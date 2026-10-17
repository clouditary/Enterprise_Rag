from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(file_path: str) -> list[dict]:
    pdf_path = Path(file_path)


    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found : {file_path}")

    reader = PdfReader(str(pdf_path))

    extracted_page = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        
        extracted_page.append({
            "page_number": page_number,
            "text": text.strip()
        })

    return extracted_page

def chunk_pdf_pages(
    pages: list[dict],
    chunk_size: int = 600,
    chunk_overlap: int = 120,
) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
        length_function=len,
    )

    chunks = []

    for page in pages:
        page_text = page.get("text", "")

        if not page_text:
            continue

        page_chunks = splitter.split_text(page_text)

        for chunk_index, chunk_text in enumerate(page_chunks):
            chunks.append({
                "chunk_id": len(chunks),
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
                "text": chunk_text,
            })

    return chunks