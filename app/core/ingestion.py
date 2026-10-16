from pathlib import Path
from pypdf import PdfReader

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

    