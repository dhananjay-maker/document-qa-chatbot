import os
import shutil
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from database import SessionLocal
from models.document_model import DocumentModel
from models.user_model import UserModel
from schemas.document import DocumentResponse, ChatRequest, ChatResponse
from services.rag_service import process_document, answer_question
from auth import get_current_user

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/upload", response_model=DocumentResponse)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_path = os.path.join(UPLOAD_DIR, f"{current_user.id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    db_doc = DocumentModel(
        filename=file.filename,
        owner_id=current_user.id,
        status="processing"
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)

    background_tasks.add_task(process_document, file_path, db_doc.id, SessionLocal())

    return db_doc


@router.get("", response_model=list[DocumentResponse])
def list_documents(
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    return db.query(DocumentModel).filter(DocumentModel.owner_id == current_user.id).all()


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    doc = (
        db.query(DocumentModel)
        .filter(DocumentModel.id == document_id, DocumentModel.owner_id == current_user.id)
        .first()
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.post("/chat", response_model=ChatResponse)
def chat_with_document(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    doc = (
        db.query(DocumentModel)
        .filter(DocumentModel.id == request.document_id, DocumentModel.owner_id == current_user.id)
        .first()
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.status != "ready":
        raise HTTPException(status_code=400, detail=f"Document is still {doc.status}. Please wait.")

    result = answer_question(request.message, doc.id, db)
    return ChatResponse(reply=result["reply"], sources=result["sources"])

@router.get("/debug/{document_id}/raw-text")
def debug_raw_text(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """TEMPORARY: shows all stored chunks for a document, in order."""
    doc = db.query(DocumentModel).filter(DocumentModel.id == document_id, DocumentModel.owner_id == current_user.id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    from models.document_chunk_model import DocumentChunkModel
    chunks = (
        db.query(DocumentChunkModel)
        .filter(DocumentChunkModel.document_id == document_id)
        .order_by(DocumentChunkModel.chunk_index)
        .all()
    )
    return {"total_chunks": len(chunks), "chunks": [c.chunk_text for c in chunks]}
    