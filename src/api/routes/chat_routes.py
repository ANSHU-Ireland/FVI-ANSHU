from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import uuid
import openai
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import json

from ...database import get_db, ChatCRUD, CompositeScoreCRUD, MetricDefinitionCRUD
from ...models import (
    ChatSessionCreate, ChatSessionResponse,
    ChatMessageCreate, ChatMessageResponse,
    MessageType
)
from ...config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Global chat sessions cache
CHAT_SESSIONS = {}


class FVIAnalystChat:
    """FVI Analyst Chat implementation with LLM integration."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory = ConversationBufferMemory(return_messages=True)
        self.llm = None
        self.context = {}
        
        # Initialize OpenAI if API key is available
        if settings.OPENAI_API_KEY:
            openai.api_key = settings.OPENAI_API_KEY
            self.llm = OpenAI(
                temperature=0.7,
                max_tokens=1000,
                openai_api_key=settings.OPENAI_API_KEY
            )
    
    def add_context(self, context: Dict[str, Any]):
        """Add context to the chat session."""
        self.context.update(context)
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for FVI analysis."""
        return """You are an expert FVI (Future Viability Index) analyst. You help users understand 
        the future viability of industries based on multiple scoring metrics including:
        
        1. Necessity Score - Social & economic indispensability
        2. Resource Scarcity Score - Difficulty and cost of resource extraction
        3. Artificial Support Score - Degree of subsidies and government support
        4. Emissions Score - GHG impact and regulatory exposure
        5. Ecological Score - Environmental externalities
        6. Workforce Transition Score - Labor market adaptability
        7. Infrastructure Repurposing Score - Asset repurposing potential
        8. Monopoly Control Score - Market structure and pricing power
        9. Economic Score - Profitability and economic resilience
        10. Technological Disruption Score - Threat from substitutes
        
        Provide clear, actionable insights based on FVI scores and component metrics. 
        Use data-driven analysis and cite specific metric values when available.
        Be concise but thorough in your explanations."""
    
    async def process_message(self, message: str, db: Session) -> Dict[str, Any]:
        """Process a chat message and generate response."""
        try:
            start_time = datetime.now()
            
            # Add user message to memory
            self.memory.chat_memory.add_user_message(message)
            
            # Analyze message intent
            intent = self._analyze_intent(message)
            
            # Get relevant context based on intent
            context = await self._get_relevant_context(intent, message, db)
            
            # Generate response
            if self.llm:
                response = await self._generate_llm_response(message, context)
            else:
                response = await self._generate_fallback_response(message, context, intent)
            
            # Add AI response to memory
            self.memory.chat_memory.add_ai_message(response["content"])
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "content": response["content"],
                "intent": intent,
                "sources_cited": response.get("sources", []),
                "confidence_score": response.get("confidence", 0.8),
                "response_time_ms": int(response_time),
                "context_used": context
            }
        
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            return {
                "content": f"I apologize, but I encountered an error processing your message: {str(e)}",
                "intent": "error",
                "sources_cited": [],
                "confidence_score": 0.0,
                "response_time_ms": 0,
                "context_used": {}
            }
    
    def _analyze_intent(self, message: str) -> str:
        """Analyze the intent of the user message."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["why", "explain", "reason"]):
            return "explain"
        elif any(word in message_lower for word in ["what if", "scenario", "change"]):
            return "scenario"
        elif any(word in message_lower for word in ["compare", "versus", "vs"]):
            return "compare"
        elif any(word in message_lower for word in ["predict", "forecast", "future"]):
            return "predict"
        elif any(word in message_lower for word in ["rank", "ranking", "top", "best", "worst"]):
            return "rank"
        elif any(word in message_lower for word in ["data", "source", "methodology"]):
            return "data_inquiry"
        else:
            return "general"
    
    async def _get_relevant_context(self, intent: str, message: str, db: Session) -> Dict[str, Any]:
        """Get relevant context based on message intent."""
        context = {}
        
        try:
            # Get recent FVI scores
            recent_scores = CompositeScoreCRUD.get_multi(db, skip=0, limit=10)
            if recent_scores:
                context["recent_scores"] = [
                    {
                        "sub_industry": score.sub_industry,
                        "country": score.country,
                        "fvi_score": score.fvi_score,
                        "horizon": score.horizon,
                        "year": score.year
                    }
                    for score in recent_scores
                ]
            
            # Get metric definitions
            if intent in ["explain", "data_inquiry"]:
                metrics = MetricDefinitionCRUD.get_multi(db, skip=0, limit=50)
                context["metrics"] = [
                    {
                        "title": metric.title,
                        "metric_slug": metric.metric_slug,
                        "details": metric.details,
                        "sheet_name": metric.sheet_name
                    }
                    for metric in metrics
                ]
            
            # Add session context
            context.update(self.context)
            
        except Exception as e:
            logger.error(f"Error getting context: {e}")
        
        return context
    
    async def _generate_llm_response(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response using LLM."""
        try:
            # Construct prompt with context
            system_prompt = self.get_system_prompt()
            context_str = json.dumps(context, indent=2)
            
            full_prompt = f"""
            {system_prompt}
            
            Context:
            {context_str}
            
            User Message: {message}
            
            Please provide a helpful response based on the FVI framework and available context.
            """
            
            response = self.llm(full_prompt)
            
            # Extract sources from context
            sources = []
            if "recent_scores" in context:
                sources.append("Recent FVI Scores")
            if "metrics" in context:
                sources.append("Metric Definitions")
            
            return {
                "content": response.strip(),
                "sources": sources,
                "confidence": 0.8
            }
        
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return await self._generate_fallback_response(message, context, "error")
    
    async def _generate_fallback_response(self, message: str, context: Dict[str, Any], intent: str) -> Dict[str, Any]:
        """Generate fallback response when LLM is not available."""
        
        if intent == "explain":
            if "recent_scores" in context and context["recent_scores"]:
                recent_score = context["recent_scores"][0]
                content = f"""Based on the most recent FVI analysis for {recent_score['sub_industry']} 
                ({recent_score['country']}), the FVI score is {recent_score['fvi_score']:.1f} 
                for a {recent_score['horizon']}-year horizon.
                
                This score reflects the industry's future viability based on 10 key metrics including 
                necessity, resource scarcity, emissions, economic factors, and technological disruption.
                
                To get a detailed explanation of specific metrics, please ask about individual 
                components like 'emissions score' or 'economic score'."""
            else:
                content = """I can help explain FVI scores and their components. FVI (Future Viability Index) 
                measures an industry's likelihood of remaining viable over 5, 10, and 20-year horizons 
                based on 10 key metrics. What specific aspect would you like me to explain?"""
        
        elif intent == "scenario":
            content = """I can help analyze scenarios for FVI scores. For example, you could ask:
            - "What if carbon tax increases by $50/ton?"
            - "How would a 20% increase in renewable energy costs affect coal's FVI?"
            - "What if workforce transition costs doubled?"
            
            Please specify the scenario you'd like to analyze."""
        
        elif intent == "compare":
            content = """I can help compare FVI scores across different industries, countries, or time periods. 
            For example:
            - Compare coal vs. renewable energy FVI scores
            - Compare coal viability across different countries
            - Compare current vs. historical FVI trends
            
            What comparison would you like to see?"""
        
        elif intent == "predict":
            content = """FVI predictions are based on machine learning models that analyze historical 
            trends and current metrics. I can help explain:
            - How FVI scores are calculated
            - What factors drive changes in FVI scores
            - Uncertainty ranges in predictions
            
            What specific prediction question do you have?"""
        
        elif intent == "data_inquiry":
            content = """FVI analysis uses data from multiple sources including:
            - International Energy Agency (IEA)
            - U.S. Energy Information Administration (EIA)
            - World Bank indicators
            - Financial market data
            - Environmental and regulatory databases
            
            What specific data source or methodology would you like to know about?"""
        
        else:
            content = """I'm your FVI analyst assistant. I can help you:
            - Understand FVI scores and their components
            - Analyze scenarios and their impacts
            - Compare industries or regions
            - Explain data sources and methodology
            - Interpret trends and predictions
            
            What would you like to explore?"""
        
        return {
            "content": content,
            "sources": ["FVI Framework"],
            "confidence": 0.6
        }


@router.post("/sessions", response_model=ChatSessionResponse)
async def create_chat_session(
    session_data: ChatSessionCreate,
    db: Session = Depends(get_db)
):
    """Create a new chat session."""
    try:
        # Generate session ID if not provided
        if not session_data.session_id:
            session_data.session_id = str(uuid.uuid4())
        
        # Create session in database
        db_session = ChatCRUD.create_session(db, session_data)
        
        # Create chat instance
        chat = FVIAnalystChat(session_data.session_id)
        CHAT_SESSIONS[session_data.session_id] = chat
        
        return db_session
    
    except Exception as e:
        logger.error(f"Error creating chat session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get chat session details."""
    try:
        session = ChatCRUD.get_session(db, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        # Get messages
        messages = ChatCRUD.get_session_messages(db, session_id)
        session.messages = messages
        
        return session
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/messages", response_model=ChatMessageResponse)
async def send_message(
    session_id: str,
    message_data: Dict[str, str],
    db: Session = Depends(get_db)
):
    """Send a message to a chat session."""
    try:
        # Check if session exists
        session = ChatCRUD.get_session(db, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        # Get or create chat instance
        if session_id not in CHAT_SESSIONS:
            CHAT_SESSIONS[session_id] = FVIAnalystChat(session_id)
        
        chat = CHAT_SESSIONS[session_id]
        
        # Create user message
        user_message = ChatMessageCreate(
            session_id=session_id,
            message_type=MessageType.USER,
            content=message_data["content"]
        )
        
        db_user_message = ChatCRUD.create_message(db, user_message)
        
        # Process message and generate response
        response = await chat.process_message(message_data["content"], db)
        
        # Create assistant message
        assistant_message = ChatMessageCreate(
            session_id=session_id,
            message_type=MessageType.ASSISTANT,
            content=response["content"],
            model_used="FVI-Analyst-v1",
            response_time_ms=response["response_time_ms"],
            sources_cited=response["sources_cited"],
            confidence_score=response["confidence_score"]
        )
        
        db_assistant_message = ChatCRUD.create_message(db, assistant_message)
        
        return db_assistant_message
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessageResponse])
async def get_session_messages(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get all messages for a chat session."""
    try:
        messages = ChatCRUD.get_session_messages(db, session_id)
        return messages
    
    except Exception as e:
        logger.error(f"Error getting session messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/context", response_model=Dict[str, Any])
async def update_session_context(
    session_id: str,
    context_data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Update chat session context."""
    try:
        # Update database
        updated_session = ChatCRUD.update_session_context(db, session_id, context_data)
        
        if not updated_session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        # Update chat instance
        if session_id in CHAT_SESSIONS:
            CHAT_SESSIONS[session_id].add_context(context_data)
        
        return {
            "message": "Context updated successfully",
            "session_id": session_id,
            "context": context_data
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating session context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/clear", response_model=Dict[str, Any])
async def clear_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Clear chat session history."""
    try:
        # Clear memory in chat instance
        if session_id in CHAT_SESSIONS:
            CHAT_SESSIONS[session_id].memory.clear()
        
        # Update session context to indicate cleared
        cleared_context = {"cleared_at": datetime.now().isoformat()}
        ChatCRUD.update_session_context(db, session_id, cleared_context)
        
        return {
            "message": "Chat session cleared successfully",
            "session_id": session_id,
            "cleared_at": cleared_context["cleared_at"]
        }
    
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/summary", response_model=Dict[str, Any])
async def get_session_summary(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get summary of chat session."""
    try:
        # Get session and messages
        session = ChatCRUD.get_session(db, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        messages = ChatCRUD.get_session_messages(db, session_id)
        
        # Calculate summary statistics
        total_messages = len(messages)
        user_messages = len([m for m in messages if m.message_type == MessageType.USER])
        assistant_messages = len([m for m in messages if m.message_type == MessageType.ASSISTANT])
        
        avg_response_time = 0
        if assistant_messages > 0:
            response_times = [m.response_time_ms for m in messages if m.response_time_ms]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Get topics discussed (simplified)
        topics = set()
        for message in messages:
            if message.message_type == MessageType.USER:
                content_lower = message.content.lower()
                if "coal" in content_lower:
                    topics.add("Coal Industry")
                if "emission" in content_lower:
                    topics.add("Emissions")
                if "economic" in content_lower:
                    topics.add("Economic Factors")
                if "scenario" in content_lower:
                    topics.add("Scenario Analysis")
        
        return {
            "session_id": session_id,
            "started_at": session.started_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "total_messages": total_messages,
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "average_response_time_ms": avg_response_time,
            "topics_discussed": list(topics),
            "is_active": session.is_active
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=Dict[str, Any])
async def chat_health_check():
    """Check chat service health."""
    return {
        "status": "healthy",
        "active_sessions": len(CHAT_SESSIONS),
        "openai_configured": settings.OPENAI_API_KEY is not None,
        "timestamp": datetime.now().isoformat()
    }
