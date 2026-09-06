import logging

logging.basicConfig(level=logging.INFO)

from fastapi import FastAPI
from routers import auth_routes, documents
from database import Base, engine
from models import document_chunk_model  # ensures table is registered with Base

app = FastAPI(title="Document Q&A Chatbot API")

Base.metadata.create_all(bind=engine)

app.include_router(auth_routes.router)
app.include_router(documents.router)

@app.get("/")
def read_root():
    return {"message": "Document Q&A Chatbot API is running"}