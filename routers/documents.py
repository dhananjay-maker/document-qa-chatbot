import os
import shutil
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from database import SessionLocal
from models.document_model import DocumentModel
from models.user_model import UserModel
from models.document_chunk_model import DocumentChunkModel
from models.chat_message_model import ChatMessageModel
from schemas.document import DocumentResponse, ChatRequest, ChatResponse, ChatMessageResponse
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


@router.delete("/{document_id}")
def delete_document(
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

    db.query(DocumentChunkModel).filter(DocumentChunkModel.document_id == document_id).delete()
    db.query(ChatMessageModel).filter(ChatMessageModel.document_id == document_id).delete()

    file_path = os.path.join(UPLOAD_DIR, f"{current_user.id}_{doc.filename}")
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass

    db.delete(doc)
    db.commit()

    return {"message": f"Document {document_id} deleted successfully"}


@router.delete("")
def delete_all_documents(
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    docs = db.query(DocumentModel).filter(DocumentModel.owner_id == current_user.id).all()

    if not docs:
        return {"message": "No documents to delete"}

    deleted_count = 0
    for doc in docs:
        db.query(DocumentChunkModel).filter(DocumentChunkModel.document_id == doc.id).delete()
        db.query(ChatMessageModel).filter(ChatMessageModel.document_id == doc.id).delete()

        file_path = os.path.join(UPLOAD_DIR, f"{current_user.id}_{doc.filename}")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass

        db.delete(doc)
        deleted_count += 1

    db.commit()

    return {"message": f"Deleted {deleted_count} document(s)"}


@router.get("/{document_id}/chat-history", response_model=list[ChatMessageResponse])
def get_chat_history(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """Returns all past chat messages for this document, in chronological order."""
    doc = (
        db.query(DocumentModel)
        .filter(DocumentModel.id == document_id, DocumentModel.owner_id == current_user.id)
        .first()
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    messages = (
        db.query(ChatMessageModel)
        .filter(ChatMessageModel.document_id == document_id, ChatMessageModel.owner_id == current_user.id)
        .order_by(ChatMessageModel.created_at.asc())
        .all()
    )
    return messages


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

    db.add(ChatMessageModel(
        document_id=request.document_id,
        owner_id=current_user.id,
        text=request.message,
        is_user=True,
        sources=[]
    ))
    db.commit()

    result = answer_question(request.message, doc.id, db)

    db.add(ChatMessageModel(
        document_id=request.document_id,
        owner_id=current_user.id,
        text=result["reply"],
        is_user=False,
        sources=result["sources"]
    ))
    db.commit()

    return ChatResponse(reply=result["reply"], sources=result["sources"])


@router.get("/debug/{document_id}/raw-text")
def debug_raw_text(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    doc = db.query(DocumentModel).filter(DocumentModel.id == document_id, DocumentModel.owner_id == current_user.id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks = (
        db.query(DocumentChunkModel)
        .filter(DocumentChunkModel.document_id == document_id)
        .order_by(DocumentChunkModel.chunk_index)
        .all()
    )
    return {"total_chunks": len(chunks), "chunks": [c.chunk_text for c in chunks]}