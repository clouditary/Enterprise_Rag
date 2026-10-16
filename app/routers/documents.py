from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pathlib import Path

router = APIRouter(
    prefix="/api/v1/documents",
    tags=["Documents"]
)

UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    department: str = Form(...)
):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed",
        )

    file_path = UPLOAD_DIR / file.filename

    content = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)

    return {
        "message": "file uploaded successfully",
        "filename": file.filename,
        "department": department
    }
