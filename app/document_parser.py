from typing import List, Dict
import os
from pathlib import Path
import pypdf
import docx2txt
import io
from tqdm import tqdm
import hashlib


class DocumentParser:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_file(self, content: bytes, filename: str) -> List[Dict]:
        """Process a file and return chunks with metadata."""
        file_extension = Path(filename).suffix.lower()
        
        if file_extension == '.pdf':
            text = self._parse_pdf(content)
        elif file_extension == '.txt':
            text = content.decode('utf-8')
        elif file_extension in ['.docx', '.doc']:
            text = self._parse_docx(content)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        file_hash = self.compute_hash(content)

        metadata = {
            "source": filename,
            "file_type": file_extension[1:],
            "file_hash": file_hash
        }
        
        chunks = self._create_chunks(text, metadata=metadata)
        
        return chunks

    def _parse_pdf(self, content: bytes) -> str:
        """Parse PDF content."""
        pdf_file = pypdf.PdfReader(io.BytesIO(content))
        text = ""
        for page in tqdm(pdf_file.pages, desc="Processing PDF"):
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text

    def _parse_docx(self, content: bytes) -> str:
        """Parse DOCX content."""
        # Save temporarily and process
        temp_path = "temp_doc.docx"
        with open(temp_path, "wb") as f:
            f.write(content)
        text = docx2txt.process(temp_path)
        os.remove(temp_path)
        return text

    def _create_chunks(self, text: str, metadata: Dict) -> List[Dict]:
        """Create overlapping chunks from text."""
        words = text.split()
        chunks = []
        
        # Calculate chunks with overlap
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            if chunk_words:
                chunk_text = " ".join(chunk_words)
                chunk_metadata = {
                    **metadata,
                    "chunk_index": len(chunks),
                    "start_index": i,
                    "end_index": i + len(chunk_words)
                }
                chunks.append({
                    "content": chunk_text,
                    "metadata": chunk_metadata
                })
        
        return chunks
    
    def compute_hash(self, content: bytes) -> str:
        """Compute SHA256 hash of the file content."""
        return hashlib.sha256(content).hexdigest()
