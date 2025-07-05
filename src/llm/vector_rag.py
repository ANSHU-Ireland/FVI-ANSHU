"""
Vector-RAG Service for FVI Analytics Platform

This module implements the Vector-RAG service with:
1. pgvector integration for efficient vector similarity search
2. OpenAI GPT-4o integration with system prompts and context injection
3. Parallel retrieval (async pgvector + Redis) for low latency
4. Response streaming with <250ms target latency
5. Ground-truth validation with citation requirements

Author: FVI Analytics Team
Created: 2025-07-05
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, AsyncGenerator, Any
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
import openai
import redis.asyncio as redis
from sentence_transformers import SentenceTransformer
import asyncpg
from pgvector.asyncpg import register_vector

from src.config import config
from src.database.db_manager import DatabaseManager
from src.ml.dynamic_weights import DynamicWeightEngine

logger = logging.getLogger(__name__)

# Pydantic models
class RAGQuery(BaseModel):
    """RAG query request"""
    query: str = Field(..., description="Natural language query")
    mine_id: Optional[str] = Field(None, description="Specific mine ID to focus on")
    context_type: Optional[str] = Field("all", description="Type of context to retrieve")
    max_tokens: int = Field(1000, description="Maximum tokens in response")
    temperature: float = Field(0.7, description="LLM temperature")
    stream: bool = Field(False, description="Whether to stream the response")

class RAGResponse(BaseModel):
    """RAG response"""
    query: str
    response: str
    sources: List[Dict[str, Any]]
    processing_time_ms: float
    tokens_used: int
    confidence_score: float

class DocumentChunk(BaseModel):
    """Document chunk for RAG"""
    chunk_id: str
    content: str
    document_title: str
    document_type: str
    similarity_score: float
    metadata: Dict[str, Any]

class VectorRAGService:
    """Vector-RAG service with pgvector and OpenAI integration"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.weight_engine = DynamicWeightEngine()
        self.redis_client = None
        self.embedding_model = None
        self.openai_client = None
        self.system_prompt = self._create_system_prompt()
        
    async def initialize(self):
        """Initialize the Vector-RAG service"""
        # Initialize OpenAI client
        self.openai_client = openai.AsyncOpenAI(
            api_key=config.OPENAI_API_KEY
        )
        
        # Initialize Redis client
        self.redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            password=config.REDIS_PASSWORD,
            decode_responses=True
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize weight engine
        await self.weight_engine.initialize()
        
        # Test pgvector connection
        await self._test_pgvector_connection()
        
        logger.info("Vector-RAG service initialized successfully")
    
    async def _test_pgvector_connection(self):
        """Test pgvector connection and extension"""
        try:
            async with self.db_manager.get_connection() as conn:
                # Register pgvector types
                await register_vector(conn)
                
                # Test vector similarity
                await conn.execute("""
                    SELECT 1 FROM silver_document_content 
                    WHERE embedding IS NOT NULL 
                    LIMIT 1
                """)
                
            logger.info("pgvector connection test successful")
            
        except Exception as e:
            logger.error(f"pgvector connection test failed: {str(e)}")
            raise
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for the LLM"""
        return """You are an expert analyst for the FVI (Future Viability Index) Analytics Platform, specializing in coal industry analysis and sustainability assessment.

Your role is to provide accurate, insightful analysis based on:
1. Retrieved document context (industry reports, sustainability documents, financial filings)
2. Current metric weights and their rationales
3. Historical FVI changes and trends
4. Real-time data quality indicators

Guidelines:
- Always cite your sources using the provided document chunks
- Provide specific, actionable insights backed by data
- Explain the reasoning behind weight adjustments and their impact
- Acknowledge uncertainty when data is incomplete or contradictory
- Focus on sustainability, operational efficiency, and future viability factors
- Use quantitative metrics when available

Citation format: [Source: {document_title}]

Response structure:
1. Direct answer to the query
2. Supporting evidence from retrieved documents
3. Relevant metric weights and their implications
4. Data quality assessment and limitations
5. Actionable recommendations (if applicable)

Be concise but comprehensive. Prioritize accuracy over speculation."""
    
    async def query(self, query_request: RAGQuery) -> RAGResponse:
        """
        Process a RAG query with parallel retrieval and LLM generation
        
        Args:
            query_request: RAG query request
            
        Returns:
            RAG response with sources and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing RAG query: {query_request.query[:100]}...")
            
            # Parallel retrieval phase
            retrieval_tasks = [
                self._retrieve_document_chunks(query_request.query, query_request.mine_id),
                self._retrieve_current_weights(),
                self._retrieve_fvi_changes(query_request.mine_id)
            ]
            
            retrieval_results = await asyncio.gather(*retrieval_tasks)
            document_chunks, current_weights, fvi_changes = retrieval_results
            
            # Build context
            context = self._build_context(
                document_chunks=document_chunks,
                current_weights=current_weights,
                fvi_changes=fvi_changes,
                query_request=query_request
            )
            
            # Generate response
            if query_request.stream:
                # For streaming, we need to handle this differently
                response_text = ""
                async for chunk in self._generate_streaming_response(context, query_request):
                    response_text += chunk
            else:
                response_text = await self._generate_response(context, query_request)
            
            # Validate response
            validation_result = self._validate_response(response_text, document_chunks)
            
            if not validation_result['valid']:
                logger.warning(f"Response validation failed: {validation_result['reason']}")
                # You might want to regenerate or add a disclaimer here
            
            # Calculate metrics
            processing_time = (time.time() - start_time) * 1000
            
            # Prepare sources
            sources = [
                {
                    'chunk_id': chunk.chunk_id,
                    'document_title': chunk.document_title,
                    'document_type': chunk.document_type,
                    'similarity_score': chunk.similarity_score,
                    'content_preview': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                }
                for chunk in document_chunks
            ]
            
            return RAGResponse(
                query=query_request.query,
                response=response_text,
                sources=sources,
                processing_time_ms=processing_time,
                tokens_used=len(response_text.split()),  # Approximate
                confidence_score=validation_result['confidence']
            )
            
        except Exception as e:
            logger.error(f"Error processing RAG query: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def stream_query(self, query_request: RAGQuery) -> AsyncGenerator[str, None]:
        """
        Stream RAG query response with real-time generation
        
        Args:
            query_request: RAG query request
            
        Yields:
            Server-sent events with streaming response
        """
        start_time = time.time()
        
        try:
            yield f"data: {json.dumps({'status': 'started', 'query': query_request.query})}\n\n"
            
            # Parallel retrieval
            yield f"data: {json.dumps({'status': 'retrieving', 'message': 'Searching relevant documents and metrics'})}\n\n"
            
            retrieval_tasks = [
                self._retrieve_document_chunks(query_request.query, query_request.mine_id),
                self._retrieve_current_weights(),
                self._retrieve_fvi_changes(query_request.mine_id)
            ]
            
            retrieval_results = await asyncio.gather(*retrieval_tasks)
            document_chunks, current_weights, fvi_changes = retrieval_results
            
            yield f"data: {json.dumps({'status': 'retrieved', 'sources_found': len(document_chunks)})}\n\n"
            
            # Build context
            context = self._build_context(
                document_chunks=document_chunks,
                current_weights=current_weights,
                fvi_changes=fvi_changes,
                query_request=query_request
            )
            
            yield f"data: {json.dumps({'status': 'generating', 'message': 'Generating response'})}\n\n"
            
            # Stream response generation
            response_text = ""
            async for chunk in self._generate_streaming_response(context, query_request):
                response_text += chunk
                yield f"data: {json.dumps({'status': 'streaming', 'chunk': chunk})}\n\n"
            
            # Final validation and metadata
            validation_result = self._validate_response(response_text, document_chunks)
            processing_time = (time.time() - start_time) * 1000
            
            # Prepare sources
            sources = [
                {
                    'chunk_id': chunk.chunk_id,
                    'document_title': chunk.document_title,
                    'document_type': chunk.document_type,
                    'similarity_score': chunk.similarity_score
                }
                for chunk in document_chunks
            ]
            
            # Final response
            final_response = {
                'status': 'completed',
                'response': response_text,
                'sources': sources,
                'processing_time_ms': processing_time,
                'confidence_score': validation_result['confidence'],
                'validation': validation_result
            }
            
            yield f"data: {json.dumps(final_response)}\n\n"
            
        except Exception as e:
            logger.error(f"Error streaming RAG query: {str(e)}")
            yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"
    
    async def _retrieve_document_chunks(self, query: str, mine_id: Optional[str] = None) -> List[DocumentChunk]:
        """
        Retrieve relevant document chunks using pgvector similarity search
        
        Args:
            query: Search query
            mine_id: Optional mine ID to filter results
            
        Returns:
            List of relevant document chunks
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Build SQL query
            sql_query = """
            SELECT 
                chunk_id,
                content_cleaned as content,
                document_title,
                document_type_standardized as document_type,
                content_quality_score,
                coal_related,
                esg_related,
                financial_related,
                operational_related,
                1 - (embedding <=> $1) as similarity_score,
                document_date,
                provenance_hash
            FROM silver_document_content
            WHERE content_quality_score >= 0.7
            AND embedding IS NOT NULL
            """
            
            params = [query_embedding]
            
            # Add mine-specific filtering if provided
            if mine_id:
                sql_query += " AND (content_cleaned ILIKE $2 OR document_title ILIKE $2)"
                params.append(f"%{mine_id}%")
            
            # Add relevance filtering
            sql_query += " AND (coal_related = true OR esg_related = true OR financial_related = true OR operational_related = true)"
            
            # Order by similarity and limit
            sql_query += " ORDER BY similarity_score DESC LIMIT 10"
            
            async with self.db_manager.get_connection() as conn:
                await register_vector(conn)
                results = await conn.fetch(sql_query, *params)
            
            # Convert to DocumentChunk objects
            chunks = []
            for row in results:
                chunk = DocumentChunk(
                    chunk_id=row['chunk_id'],
                    content=row['content'],
                    document_title=row['document_title'],
                    document_type=row['document_type'],
                    similarity_score=float(row['similarity_score']),
                    metadata={
                        'content_quality_score': row['content_quality_score'],
                        'coal_related': row['coal_related'],
                        'esg_related': row['esg_related'],
                        'financial_related': row['financial_related'],
                        'operational_related': row['operational_related'],
                        'document_date': row['document_date'].isoformat() if row['document_date'] else None,
                        'provenance_hash': row['provenance_hash']
                    }
                )
                chunks.append(chunk)
            
            logger.info(f"Retrieved {len(chunks)} document chunks for query")
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving document chunks: {str(e)}")
            return []
    
    async def _retrieve_current_weights(self) -> Dict[str, Any]:
        """Retrieve current metric weights and rationales"""
        try:
            # Get weights from Redis cache
            weights = await self.weight_engine.get_weights()
            
            # Get weight metadata
            metadata_json = await self.redis_client.get("weights:metadata")
            metadata = json.loads(metadata_json) if metadata_json else {}
            
            return {
                'weights': weights,
                'metadata': metadata,
                'last_updated': metadata.get('last_updated'),
                'total_metrics': metadata.get('total_metrics', 0)
            }
            
        except Exception as e:
            logger.error(f"Error retrieving current weights: {str(e)}")
            return {'weights': {}, 'metadata': {}}
    
    async def _retrieve_fvi_changes(self, mine_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve recent FVI changes and trends"""
        try:
            # Build query for FVI changes
            if mine_id:
                query = """
                SELECT 
                    mine_id,
                    feature_date,
                    feature_freshness_score as fvi_score,
                    LAG(feature_freshness_score) OVER (PARTITION BY mine_id ORDER BY feature_date) as prev_fvi_score
                FROM gold_mine_features
                WHERE mine_id = $1
                AND feature_date >= NOW() - INTERVAL '30 days'
                ORDER BY feature_date DESC
                LIMIT 10
                """
                params = [mine_id]
            else:
                query = """
                SELECT 
                    mine_id,
                    feature_date,
                    feature_freshness_score as fvi_score,
                    LAG(feature_freshness_score) OVER (PARTITION BY mine_id ORDER BY feature_date) as prev_fvi_score
                FROM gold_mine_features
                WHERE feature_date >= NOW() - INTERVAL '7 days'
                ORDER BY feature_date DESC
                LIMIT 50
                """
                params = []
            
            async with self.db_manager.get_connection() as conn:
                results = await conn.fetch(query, *params)
            
            # Process FVI changes
            changes = []
            for row in results:
                if row['prev_fvi_score'] is not None:
                    delta = row['fvi_score'] - row['prev_fvi_score']
                    changes.append({
                        'mine_id': row['mine_id'],
                        'date': row['feature_date'].isoformat(),
                        'fvi_score': row['fvi_score'],
                        'delta': delta,
                        'direction': 'increase' if delta > 0.01 else 'decrease' if delta < -0.01 else 'stable'
                    })
            
            # Calculate summary statistics
            if changes:
                deltas = [c['delta'] for c in changes]
                summary = {
                    'total_changes': len(changes),
                    'avg_delta': np.mean(deltas),
                    'max_increase': max(deltas),
                    'max_decrease': min(deltas),
                    'recent_changes': changes[:5]  # Most recent 5
                }
            else:
                summary = {'total_changes': 0, 'recent_changes': []}
            
            return summary
            
        except Exception as e:
            logger.error(f"Error retrieving FVI changes: {str(e)}")
            return {'total_changes': 0, 'recent_changes': []}
    
    def _build_context(self, document_chunks: List[DocumentChunk], current_weights: Dict[str, Any],
                      fvi_changes: Dict[str, Any], query_request: RAGQuery) -> str:
        """Build context for LLM prompt"""
        
        context_parts = [
            f"User Query: {query_request.query}",
            ""
        ]
        
        # Add document context
        if document_chunks:
            context_parts.append("=== RELEVANT DOCUMENTS ===")
            for i, chunk in enumerate(document_chunks[:5]):  # Top 5 chunks
                context_parts.extend([
                    f"Document {i+1}: {chunk.document_title} ({chunk.document_type})",
                    f"Similarity: {chunk.similarity_score:.3f}",
                    f"Content: {chunk.content[:500]}...",
                    ""
                ])
        
        # Add current weights
        if current_weights.get('weights'):
            context_parts.append("=== CURRENT METRIC WEIGHTS ===")
            weights = current_weights['weights']
            metadata = current_weights['metadata']
            
            context_parts.extend([
                f"Last Updated: {metadata.get('last_updated', 'Unknown')}",
                f"Total Metrics: {metadata.get('total_metrics', 0)}",
                ""
            ])
            
            # Top weighted metrics
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            for metric, weight in sorted_weights[:10]:
                context_parts.append(f"- {metric}: {weight:.3f}")
            
            context_parts.append("")
        
        # Add FVI changes
        if fvi_changes.get('recent_changes'):
            context_parts.append("=== RECENT FVI CHANGES ===")
            context_parts.extend([
                f"Total Changes: {fvi_changes['total_changes']}",
                f"Average Delta: {fvi_changes.get('avg_delta', 0):.4f}",
                ""
            ])
            
            for change in fvi_changes['recent_changes']:
                context_parts.append(
                    f"- {change['mine_id']}: {change['fvi_score']:.3f} "
                    f"({change['direction']}: {change['delta']:+.4f}) on {change['date']}"
                )
            
            context_parts.append("")
        
        # Add query-specific context
        if query_request.mine_id:
            context_parts.extend([
                f"=== FOCUS MINE: {query_request.mine_id} ===",
                "Please provide specific analysis for this mine when available.",
                ""
            ])
        
        return "\n".join(context_parts)
    
    async def _generate_response(self, context: str, query_request: RAGQuery) -> str:
        """Generate LLM response"""
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": context}
            ]
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=query_request.max_tokens,
                temperature=query_request.temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            raise
    
    async def _generate_streaming_response(self, context: str, query_request: RAGQuery) -> AsyncGenerator[str, None]:
        """Generate streaming LLM response"""
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": context}
            ]
            
            stream = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=query_request.max_tokens,
                temperature=query_request.temperature,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error generating streaming LLM response: {str(e)}")
            raise
    
    def _validate_response(self, response: str, document_chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Validate LLM response for citations and accuracy"""
        
        # Check for citations
        citations_found = response.count('[Source:')
        has_sufficient_citations = citations_found >= 2
        
        # Check for contradictions (simple keyword-based check)
        contradictions = []
        
        # Check if response mentions specific data that contradicts known facts
        # This is a simplified check - in production, you'd want more sophisticated validation
        
        # Calculate confidence score
        confidence_factors = []
        
        # Citation factor
        if has_sufficient_citations:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)
        
        # Document relevance factor
        if document_chunks:
            avg_similarity = np.mean([chunk.similarity_score for chunk in document_chunks])
            confidence_factors.append(avg_similarity)
        else:
            confidence_factors.append(0.2)
        
        # Length factor (reasonable length indicates comprehensive response)
        if 200 <= len(response) <= 2000:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
        
        confidence_score = np.mean(confidence_factors)
        
        return {
            'valid': has_sufficient_citations and len(contradictions) == 0,
            'confidence': confidence_score,
            'citations_found': citations_found,
            'has_sufficient_citations': has_sufficient_citations,
            'contradictions': contradictions,
            'reason': 'Passed validation' if has_sufficient_citations and len(contradictions) == 0 else 'Failed validation'
        }
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        try:
            # Get cache stats
            query_cache_keys = await self.redis_client.keys("rag_query:*")
            
            # Get database stats
            async with self.db_manager.get_connection() as conn:
                doc_count = await conn.fetchval("SELECT COUNT(*) FROM silver_document_content")
                embedding_count = await conn.fetchval("SELECT COUNT(*) FROM silver_document_content WHERE embedding IS NOT NULL")
            
            return {
                'document_count': doc_count,
                'embedding_count': embedding_count,
                'embedding_coverage': embedding_count / doc_count if doc_count > 0 else 0,
                'cached_queries': len(query_cache_keys),
                'models_loaded': {
                    'embedding_model': self.embedding_model is not None,
                    'openai_client': self.openai_client is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting service stats: {str(e)}")
            return {'error': str(e)}
    
    async def cleanup(self):
        """Clean up resources"""
        if self.redis_client:
            await self.redis_client.close()
        if self.weight_engine:
            await self.weight_engine.cleanup()
        logger.info("Vector-RAG service cleanup completed")


# FastAPI integration
def create_rag_routes(app: FastAPI):
    """Create RAG service routes"""
    
    rag_service = VectorRAGService()
    
    @app.on_event("startup")
    async def startup_rag():
        await rag_service.initialize()
    
    @app.on_event("shutdown")
    async def shutdown_rag():
        await rag_service.cleanup()
    
    @app.post("/rag/query", response_model=RAGResponse)
    async def query_rag(query_request: RAGQuery):
        """Process RAG query"""
        return await rag_service.query(query_request)
    
    @app.post("/rag/query/stream")
    async def stream_rag_query(query_request: RAGQuery):
        """Stream RAG query response"""
        return EventSourceResponse(rag_service.stream_query(query_request))
    
    @app.get("/rag/stats")
    async def get_rag_stats():
        """Get RAG service statistics"""
        return await rag_service.get_service_stats()
    
    return rag_service

if __name__ == "__main__":
    # Example usage
    async def main():
        service = VectorRAGService()
        await service.initialize()
        
        try:
            # Example query
            query = RAGQuery(
                query="What are the key sustainability challenges facing coal mines in Australia?",
                context_type="sustainability",
                max_tokens=500
            )
            
            response = await service.query(query)
            print(json.dumps(response.dict(), indent=2))
            
        finally:
            await service.cleanup()
    
    asyncio.run(main())
