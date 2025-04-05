from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from pathlib import Path
import logging
import os
from pydantic import BaseModel

from .document_parser import DocumentParser
from .retriever import Retriever
from .llm import LocalLLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="RAG Prototype API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get model path from environment variable or use default
MODEL_NAME = os.getenv("MODEL_NAME", "mistral-7b-instruct-v0.2")
USE_4BIT = os.getenv("USE_4BIT", "True").lower() == "true"

# Initialize components
doc_parser = DocumentParser()
retriever = Retriever()
llm = LocalLLM(
    model_name=MODEL_NAME,
    use_4bit=USE_4BIT
)

# Pydantic models
class Query(BaseModel):
    query: str
    top_k: Optional[int] = 5

class Response(BaseModel):
    answer: str
    sources: List[Dict]

class ModelConfig(BaseModel):
    model_name: str
    use_4bit: bool

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document."""
    try:
        content = await file.read()

        file_hash = doc_parser.compute_hash(content)

        if retriever.file_already_exists(file_hash):
            return {
                "message": f"Duplicate file skipped: {file.filename}",
                "filename": file.filename,
                "chunks": 0
            }
        
        chunks = doc_parser.process_file(content, file.filename)
        retriever.add_documents(chunks)
        
        return {
            "message": "Document processed successfully",
            "filename": file.filename,
            "chunks": len(chunks)
        }
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=Response)
async def query(query: Query):
    """Process a query and return response with sources."""
    try:
        # Retrieve relevant documents
        relevant_chunks = retriever.retrieve(
            query.query,
            rerank_k=query.top_k
        )
        
        # Generate response using LLM
        response = llm.generate_response(
            query.query,
            relevant_chunks
        )
        
        return response
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """Get system statistics."""
    try:
        return retriever.list_documents()
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class DeleteRequest(BaseModel):
    filename: str


@app.delete("/documents/delete")
async def delete_document(req: DeleteRequest):
    try:
        retriever.delete_by_filename(req.filename)
        return {"message": f"Deleted all chunks from {req.filename}"}
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    try:
        return {
            "model_name": MODEL_NAME,
            "use_4bit": USE_4BIT,
            "device": llm.device
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload-model")
async def reload_model(config: ModelConfig):
    """Reload the model with new configuration."""
    try:
        global llm
        llm = LocalLLM(
            model_name=config.model_name,
            use_4bit=config.use_4bit
        )
        return {
            "message": "Model reloaded successfully",
            "model_name": config.model_name,
            "use_4bit": config.use_4bit
        }
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True) 