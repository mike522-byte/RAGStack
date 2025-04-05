from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from pathlib import Path
import uuid

class Retriever:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        persist_directory: str = "data/vectorstore"
    ):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize reranker
        self.reranker = CrossEncoder(reranker_model)
        
        # Initialize ChromaDB
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[Dict]) -> None:
        """Add documents to the vector store."""
        # Prepare documents for ChromaDB
        texts = [doc["content"] for doc in documents]
        ids = [str(uuid.uuid4()) for _ in documents]
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        ).tolist()
        metadatas = [doc["metadata"] for doc in documents]
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

    def retrieve(self, query: str, k: int = 10, rerank_k: int = 5) -> List[Dict]:
        """Retrieve relevant documents using hybrid search."""
        # Get embeddings for query
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True
        ).tolist()
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Prepare documents for reranking
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        
        # Rerank using cross-encoder
        rerank_pairs = [(query, doc) for doc in documents]
        rerank_scores = self.reranker.predict(rerank_pairs)
        
        # Combine results
        ranked_results = []
        for i in range(len(documents)):
            ranked_results.append({
                "content": documents[i],
                "metadata": metadatas[i],
                "similarity_score": 1 - results["distances"][0][i],  # Convert distance to similarity
                "rerank_score": float(rerank_scores[i])
            })
        
        # Sort by rerank score and return top k
        ranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return ranked_results[:rerank_k]

    def list_documents(self) -> List[Dict]:
        """List all documents with their source filenames and chunk counts."""
        try:
            result = self.collection.get(include=["metadatas"])
            metadatas = result.get("metadatas", [])
            
            doc_chunks = {}
            for meta in metadatas:
                if not meta:
                    continue
                source = meta.get("source", "Unknown")
                doc_chunks[source] = doc_chunks.get(source, 0) + 1

            return [
                {"filename": name, "chunk_count": count}
                for name, count in doc_chunks.items()
            ]
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e
    
    def delete_by_filename(self, filename: str):
        """Delete all chunks associated with a given filename."""
        try:
            # Get all documents with their IDs and metadata
            results = self.collection.get(
                where={"source": filename}
            )
            
            # Convert to set to remove duplicates, then back to list
            unique_ids = list(set(results["ids"]))
            
            if unique_ids:
                self.collection.delete(ids=unique_ids)
        except Exception as e:
            print(f"Error deleting document: {str(e)}")
            raise
    
    def file_already_exists(self, file_hash: str) -> bool:
        """Check if a document with the same hash already exists."""
        results = self.collection.get(include=["metadatas"])
        for meta in results.get("metadatas", []):
            if meta and meta.get("file_hash") == file_hash:
                return True
        return False


    
    