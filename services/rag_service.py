import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from sqlalchemy.orm import Session

from models.document_model import DocumentModel
from models.document_chunk_model import DocumentChunkModel

load_dotenv()
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


def extract_text_from_pdf(file_path: str) -> str:
    text_parts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    for row in table:
                        if not any(cell for cell in row):
                            continue
                        row_text = " | ".join(cell.strip() if cell else "" for cell in row)
                        text_parts.append(row_text)
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    return [c for c in chunks if len(c.strip()) > 10]


def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def process_document(file_path: str, document_id: int, db: Session):
    try:
        raw_text = extract_text_from_pdf(file_path)
        chunks = chunk_text(raw_text)

        if not chunks:
            raise ValueError("No extractable text found in this PDF.")

        for index, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            db_chunk = DocumentChunkModel(
                document_id=document_id,
                chunk_text=chunk,
                chunk_index=index,
                embedding=embedding
            )
            db.add(db_chunk)

        document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
        document.status = "ready"
        db.commit()

        logger.info(f"Document {document_id} processed: {len(chunks)} chunks stored in PostgreSQL.")

    except Exception as e:
        logger.error(f"Failed to process document {document_id}: {e}")
        document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
        if document:
            document.status = "failed"
            db.commit()
        raise
    finally:
        db.close()


def _retrieve_raw(query: str, document_id: int, db: Session, k: int = 12):
    embedding = get_embedding(query)
    return (
        db.query(DocumentChunkModel)
        .filter(DocumentChunkModel.document_id == document_id)
        .order_by(DocumentChunkModel.embedding.cosine_distance(embedding))
        .limit(k)
        .all()
    )


def answer_question(question: str, document_id: int, db: Session) -> dict:
    """
    Maximally robust retrieval pipeline: forced initial retrieval (k=12,
    increased from 8 for better recall), agent can search again if needed,
    and code-level retry escalates to the FULL document if the answer
    still comes back empty. This minimizes (but cannot fully eliminate)
    missed answers — see class docstring note on RAG's inherent limits.
    """
    retrieved_sources = []

    initial_results = _retrieve_raw(question, document_id, db, k=12)
    for r in initial_results:
        snippet = r.chunk_text[:200] + "..."
        if snippet not in retrieved_sources:
            retrieved_sources.append(snippet)

    initial_context = (
        "\n\n---\n\n".join(r.chunk_text for r in initial_results)
        if initial_results else "No relevant passages found."
    )

    @tool
    def retrieve_context(query: str) -> str:
        """Search the document again with a different or more specific query,
        if the initial context wasn't sufficient (e.g. multi-part questions
        where only some items were covered by the first search)."""
        logger.info(f"[TOOL CALLED] retrieve_context(query={query!r})")
        try:
            results = _retrieve_raw(query, document_id, db, k=12)
            logger.info(f"[TOOL RESULT] found {len(results)} chunks")

            if not results:
                return "No relevant passages found for this query."

            for r in results:
                snippet = r.chunk_text[:200] + "..."
                if snippet not in retrieved_sources:
                    retrieved_sources.append(snippet)

            return "\n\n---\n\n".join(r.chunk_text for r in results)
        except Exception as e:
            logger.error(f"[TOOL ERROR] {e}", exc_info=True)
            return f"Search error occurred: {e}"

    system_message = (
        "You are a document assistant. Below is INITIAL CONTEXT already "
        "retrieved from the uploaded document for this question.\n\n"
        f"INITIAL CONTEXT:\n{initial_context}\n\n"
        "If this context fully answers the question, respond directly using "
        "ONLY this context — do not call any tool.\n\n"
        "If the question asks about MULTIPLE distinct items/values and this "
        "context only covers some of them, call retrieve_context with a "
        "specific query for EACH missing item before answering.\n\n"
        "CRITICAL VERIFICATION RULE: Before including ANY specific fact, "
        "number, date, or claim, verify it appears explicitly in the "
        "retrieved context. Never supplement with outside/training "
        "knowledge, even facts you are confident about. If something isn't "
        "found after searching, say so clearly rather than guessing.\n\n"
        "Keep answers concise (2-4 sentences) unless asked for detail. Never "
        "dump raw data or chunks, and never write code not present in the "
        "document. For medical documents, never suggest contacting lab "
        "staff as doctors — recommend a real healthcare professional. If "
        "clearly unrelated to any document, say you can only answer "
        "document questions."
    )

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    agent = create_react_agent(model=llm, tools=[retrieve_context])

    try:
        response = agent.invoke({
            "messages": [
                SystemMessage(content=system_message),
                HumanMessage(content=question)
            ]
        })
        answer = response["messages"][-1].content
    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        return {
            "reply": "Sorry, something went wrong while processing your question. Please try again.",
            "sources": []
        }

    # CODE-LEVEL RETRY: if the answer indicates nothing was found, escalate
    # to searching the ENTIRE document as a last resort before giving up.
    not_found_phrases = ["not found", "does not provide", "couldn't find", "no information", "not mentioned", "not available"]
    if any(phrase in answer.lower() for phrase in not_found_phrases):
        logger.info("Answer indicates 'not found' — escalating to full-document retry.")
        all_chunks = (
            db.query(DocumentChunkModel)
            .filter(DocumentChunkModel.document_id == document_id)
            .order_by(DocumentChunkModel.chunk_index)
            .all()
        )
        full_context = "\n\n---\n\n".join(c.chunk_text for c in all_chunks[:80])

        retry_messages = [
            SystemMessage(content=(
                "Answer using ONLY this full document context. Be concise. "
                "Only say 'not found' if you have genuinely checked all of it.\n\n"
                f"FULL DOCUMENT:\n{full_context}"
            )),
            HumanMessage(content=question)
        ]
        try:
            retry_response = llm.invoke(retry_messages)
            retry_answer = retry_response.content
            if not any(phrase in retry_answer.lower() for phrase in not_found_phrases):
                answer = retry_answer
                for c in all_chunks[:10]:
                    snippet = c.chunk_text[:200] + "..."
                    if snippet not in retrieved_sources:
                        retrieved_sources.append(snippet)
        except Exception as e:
            logger.warning(f"Full-document retry failed: {e}")

    return {"reply": answer, "sources": retrieved_sources[:10]}