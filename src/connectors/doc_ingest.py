"""
Document Ingestion Pipeline
==========================

Pipeline for chunking unstructured PDFs into markdown + embeddings.
Includes provenance hash tracking and content versioning.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import hashlib
import json
from datetime import datetime
import uuid

# PDF processing
import PyPDF2
import pdfplumber
from pdfminer.high_level import extract_text

# Text processing
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Markdown conversion
import markdownify
from bs4 import BeautifulSoup

# Database
import asyncpg
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a processed document chunk."""
    chunk_id: str
    document_id: str
    content: str
    markdown: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    provenance_hash: str
    created_at: datetime


class DocumentProcessor:
    """Processes PDFs into structured chunks with embeddings."""
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize NLP models
        self.nlp = None
        self.tokenizer = None
        self.model = None
        
    async def initialize(self):
        """Initialize NLP models."""
        logger.info("Initializing NLP models...")
        
        # Load spaCy for text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
            
        # Load embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
        self.model = AutoModel.from_pretrained(self.embedding_model)
        
        logger.info("NLP models initialized")
        
    def generate_document_id(self, file_path: Path) -> str:
        """Generate unique document ID."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(file_path)))
        
    def generate_provenance_hash(self, content: str, metadata: Dict) -> str:
        """Generate provenance hash for content tracking."""
        combined = content + json.dumps(metadata, sort_keys=True)
        return hashlib.sha256(combined.encode()).hexdigest()
        
    def extract_text_from_pdf(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract text and metadata from PDF."""
        text_content = ""
        metadata = {
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'pages': 0,
            'extraction_method': 'hybrid'
        }
        
        try:
            # Try pdfplumber first (better for tables and complex layouts)
            with pdfplumber.open(file_path) as pdf:
                metadata['pages'] = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += f"\n\n--- Page {page_num + 1} ---\n\n"
                            text_content += page_text
                            
                        # Extract tables if present
                        tables = page.extract_tables()
                        if tables:
                            for table_num, table in enumerate(tables):
                                text_content += f"\n\n--- Table {table_num + 1} on Page {page_num + 1} ---\n\n"
                                for row in table:
                                    if row:
                                        text_content += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
                                        
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num}: {e}")
                        
        except Exception as e:
            logger.warning(f"pdfplumber failed, trying PyPDF2: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    metadata['pages'] = len(pdf_reader.pages)
                    metadata['extraction_method'] = 'pypdf2'
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text_content += f"\n\n--- Page {page_num + 1} ---\n\n"
                                text_content += page_text
                        except Exception as e:
                            logger.warning(f"Failed to extract page {page_num}: {e}")
                            
            except Exception as e:
                logger.error(f"All PDF extraction methods failed: {e}")
                raise
                
        # Clean up text
        text_content = self.clean_text(text_content)
        metadata['character_count'] = len(text_content)
        metadata['extracted_at'] = datetime.utcnow().isoformat()
        
        return text_content, metadata
        
    def clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                cleaned_lines.append(line)
                
        # Rejoin with single newlines
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove excessive spaces
        import re
        cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)
        
        return cleaned_text
        
    def convert_to_markdown(self, text: str, metadata: Dict) -> str:
        """Convert text to markdown format."""
        markdown = f"# {metadata.get('title', 'Document')}\n\n"
        
        if metadata.get('file_path'):
            markdown += f"**Source:** {metadata['file_path']}\n"
        if metadata.get('extracted_at'):
            markdown += f"**Extracted:** {metadata['extracted_at']}\n"
        if metadata.get('pages'):
            markdown += f"**Pages:** {metadata['pages']}\n"
            
        markdown += "\n---\n\n"
        
        # Process text into sections
        lines = text.split('\n')
        current_section = []
        
        for line in lines:
            if line.startswith('--- Page'):
                if current_section:
                    markdown += '\n'.join(current_section) + '\n\n'
                    current_section = []
                markdown += f"## {line.strip(' -')}\n\n"
            elif line.startswith('--- Table'):
                if current_section:
                    markdown += '\n'.join(current_section) + '\n\n'
                    current_section = []
                markdown += f"### {line.strip(' -')}\n\n"
            else:
                # Detect potential headers (ALL CAPS lines or numbered sections)
                if (len(line) < 100 and 
                    (line.isupper() or 
                     (line[0].isdigit() and '.' in line[:10]))):
                    if current_section:
                        markdown += '\n'.join(current_section) + '\n\n'
                        current_section = []
                    markdown += f"### {line}\n\n"
                else:
                    current_section.append(line)
                    
        # Add remaining content
        if current_section:
            markdown += '\n'.join(current_section) + '\n\n'
            
        return markdown
        
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text into overlapping segments."""
        if not self.tokenizer:
            # Simple word-based chunking if no tokenizer
            words = text.split()
            chunks = []
            
            for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                chunk_words = words[i:i + self.chunk_size]
                chunks.append(' '.join(chunk_words))
                
            return chunks
            
        # Token-based chunking
        tokens = self.tokenizer.tokenize(text)
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)
            
        return chunks
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks."""
        if not self.model or not self.tokenizer:
            logger.warning("Embedding model not initialized, returning zeros")
            return np.zeros((len(texts), 384))  # Default dimension
            
        embeddings = []
        
        for text in texts:
            try:
                # Tokenize and encode
                inputs = self.tokenizer(text, 
                                      return_tensors="pt", 
                                      truncation=True, 
                                      padding=True, 
                                      max_length=512)
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use mean pooling
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                    
                embeddings.append(embedding)
                
            except Exception as e:
                logger.warning(f"Failed to generate embedding for text chunk: {e}")
                # Use zero vector as fallback
                embeddings.append(np.zeros(384))
                
        return np.array(embeddings)
        
    async def process_document(self, file_path: Path) -> List[DocumentChunk]:
        """Process a PDF document into chunks with embeddings."""
        logger.info(f"Processing document: {file_path}")
        
        try:
            # Extract text and metadata
            text_content, metadata = self.extract_text_from_pdf(file_path)
            
            if not text_content.strip():
                logger.warning(f"No text extracted from {file_path}")
                return []
                
            # Generate document ID
            document_id = self.generate_document_id(file_path)
            
            # Convert to markdown
            markdown_content = self.convert_to_markdown(text_content, metadata)
            
            # Chunk the text
            text_chunks = self.chunk_text(text_content)
            markdown_chunks = self.chunk_text(markdown_content)
            
            # Ensure same number of chunks
            min_chunks = min(len(text_chunks), len(markdown_chunks))
            text_chunks = text_chunks[:min_chunks]
            markdown_chunks = markdown_chunks[:min_chunks]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(text_chunks)
            
            # Create DocumentChunk objects
            chunks = []
            for i, (text_chunk, markdown_chunk, embedding) in enumerate(
                zip(text_chunks, markdown_chunks, embeddings)
            ):
                chunk_metadata = {
                    **metadata,
                    'chunk_index': i,
                    'total_chunks': len(text_chunks),
                    'chunk_size': len(text_chunk),
                }
                
                chunk = DocumentChunk(
                    chunk_id=f"{document_id}_{i:04d}",
                    document_id=document_id,
                    content=text_chunk,
                    markdown=markdown_chunk,
                    embedding=embedding,
                    metadata=chunk_metadata,
                    provenance_hash=self.generate_provenance_hash(text_chunk, chunk_metadata),
                    created_at=datetime.utcnow()
                )
                
                chunks.append(chunk)
                
            logger.info(f"Created {len(chunks)} chunks for {file_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {e}")
            return []


class DocumentStore:
    """Stores processed documents in PostgreSQL with pgvector."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
        
    async def initialize(self):
        """Initialize database connection pool."""
        self.pool = await asyncpg.create_pool(self.database_url)
        await self.create_tables()
        
    async def create_tables(self):
        """Create tables for document storage."""
        async with self.pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create documents table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    title TEXT,
                    metadata JSONB,
                    processed_at TIMESTAMP DEFAULT NOW(),
                    chunk_count INTEGER DEFAULT 0
                )
            """)
            
            # Create document_chunks table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id UUID REFERENCES documents(id),
                    content TEXT NOT NULL,
                    markdown TEXT,
                    embedding VECTOR(384),
                    metadata JSONB,
                    provenance_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id 
                ON document_chunks(document_id)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding 
                ON document_chunks USING ivfflat (embedding vector_cosine_ops)
            """)
            
    async def store_document(self, chunks: List[DocumentChunk]) -> bool:
        """Store document chunks in database."""
        if not chunks:
            return False
            
        try:
            async with self.pool.acquire() as conn:
                # Store document record
                document_id = chunks[0].document_id
                metadata = chunks[0].metadata
                
                await conn.execute("""
                    INSERT INTO documents (id, file_path, title, metadata, chunk_count)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (id) DO UPDATE SET
                        metadata = EXCLUDED.metadata,
                        chunk_count = EXCLUDED.chunk_count,
                        processed_at = NOW()
                """, 
                document_id, 
                metadata.get('file_path'),
                metadata.get('title', 'Untitled'),
                json.dumps(metadata),
                len(chunks)
                )
                
                # Store chunks
                for chunk in chunks:
                    await conn.execute("""
                        INSERT INTO document_chunks 
                        (chunk_id, document_id, content, markdown, embedding, metadata, provenance_hash)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            content = EXCLUDED.content,
                            markdown = EXCLUDED.markdown,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata,
                            provenance_hash = EXCLUDED.provenance_hash,
                            created_at = NOW()
                    """,
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.content,
                    chunk.markdown,
                    chunk.embedding.tolist(),  # Convert numpy array to list
                    json.dumps(chunk.metadata),
                    chunk.provenance_hash
                    )
                    
            logger.info(f"Stored {len(chunks)} chunks for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store document: {e}")
            return False
            
    async def search_similar(self, query_embedding: np.ndarray, limit: int = 10) -> List[Dict]:
        """Search for similar document chunks."""
        try:
            async with self.pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT chunk_id, document_id, content, markdown, metadata,
                           embedding <-> $1 as distance
                    FROM document_chunks
                    ORDER BY embedding <-> $1
                    LIMIT $2
                """, query_embedding.tolist(), limit)
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
            
    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()


class DocumentIngestionPipeline:
    """Main pipeline for document ingestion."""
    
    def __init__(self, database_url: str, input_dir: str = "data/documents"):
        self.input_dir = Path(input_dir)
        self.processor = DocumentProcessor()
        self.store = DocumentStore(database_url)
        
    async def initialize(self):
        """Initialize pipeline components."""
        await self.processor.initialize()
        await self.store.initialize()
        
    async def process_directory(self, file_pattern: str = "*.pdf") -> Dict[str, Any]:
        """Process all PDFs in input directory."""
        pdf_files = list(self.input_dir.glob(file_pattern))
        
        results = {
            'processed': 0,
            'failed': 0,
            'total_chunks': 0,
            'files': []
        }
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing {pdf_file}")
                
                chunks = await self.processor.process_document(pdf_file)
                if chunks:
                    success = await self.store.store_document(chunks)
                    if success:
                        results['processed'] += 1
                        results['total_chunks'] += len(chunks)
                        results['files'].append({
                            'file': str(pdf_file),
                            'chunks': len(chunks),
                            'status': 'success'
                        })
                    else:
                        results['failed'] += 1
                        results['files'].append({
                            'file': str(pdf_file),
                            'chunks': 0,
                            'status': 'storage_failed'
                        })
                else:
                    results['failed'] += 1
                    results['files'].append({
                        'file': str(pdf_file),
                        'chunks': 0,
                        'status': 'processing_failed'
                    })
                    
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
                results['failed'] += 1
                results['files'].append({
                    'file': str(pdf_file),
                    'chunks': 0,
                    'status': 'error',
                    'error': str(e)
                })
                
        return results
        
    async def close(self):
        """Close pipeline resources."""
        await self.store.close()


async def main():
    """Main function for testing document ingestion."""
    pipeline = DocumentIngestionPipeline("postgresql://fvi_user:fvi_password@localhost:5432/fvi_db")
    
    try:
        await pipeline.initialize()
        results = await pipeline.process_directory()
        
        print(f"Processing complete:")
        print(f"  Processed: {results['processed']} files")
        print(f"  Failed: {results['failed']} files") 
        print(f"  Total chunks: {results['total_chunks']}")
        
    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
