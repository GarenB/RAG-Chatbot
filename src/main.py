import json
import uuid
from pathlib import Path
from sse_starlette.sse import EventSourceResponse
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.rag.pipeline import RAGPipeline

load_dotenv()

rag_pipeline = RAGPipeline()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing RAG pipeline...")
    rag_pipeline.initialize()
    print("RAG pipeline ready!")
    yield
    rag_pipeline.weaviate_client.close()
    print("Shutting down application...")


app = FastAPI(
    title="RAG Knowledge Assistant API",
    description="A document-based question answering API powered by Ollama and FAISS.",
    version="1.0.0",
    lifespan=lifespan
)

class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

@app.get("/")
def home():
    return {
        "message": "RAG Knowledge Assistant API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        response = rag_pipeline.ask(request.question, session_id="user_1")
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/ask/stream")
async def ask_question_stream(request: QuestionRequest):
    async def event_generator():
        try:
            async for event in rag_pipeline.ask_stream(request.question, session_id="user_1"):
                yield {
                    "event": event["event"],
                    "data": json.dumps(event["data"])
                }
        except Exception:
            yield {
                "event": "error",
                "data": json.dumps({
                    "message": "An error occurred during generation."
                })
            }
    return EventSourceResponse(event_generator())

@app.post("/upload")
async def upload_pdf(file:UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )
        
    file_path = UPLOAD_DIR / f"{uuid.uuid4()}.pdf"
    
    try:
        content = await file.read()
        if not content.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=400,
                detail="Invalid PDF file"
            )
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="PDF must be smaller than 20 MB."
            )
        
        file_path.write_bytes(content)
        cnt_chunks = rag_pipeline.upload_pdf(str(file_path), file.filename)
        return {
            "message": "PDF uploaded and indexed successfully.",
            "filename": file.filename,
            "chunks": cnt_chunks
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    finally:
        file_path.unlink(missing_ok=True)

@app.get("/documents")
def get_uploaded_documents():
    try:
        return rag_pipeline.get_uploaded_documents()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )