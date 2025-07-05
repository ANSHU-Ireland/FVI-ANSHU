#!/usr/bin/env python3
"""
Simple FastAPI server for FVI Analytics Platform
Quick start version with minimal dependencies
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from datetime import datetime
import json

# Create FastAPI app
app = FastAPI(
    title="FVI Analytics Platform",
    description="Future Viability Index Analytics Platform - Simple Version",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data for demo
MOCK_METRICS = {
    "total_metrics": 46,
    "active_mines": 8,
    "avg_trust_score": 0.85,
    "last_updated": datetime.now().isoformat()
}

MOCK_CHAT_RESPONSES = [
    "Based on our coal market analysis, the global demand has shifted significantly due to renewable energy adoption.",
    "The FVI scoring model indicates that mines with strong ESG profiles show 23% better long-term viability.",
    "Current data suggests that mines in regions with supportive policy frameworks have higher sustainability scores.",
    "Environmental impact assessments show that modern mining techniques can reduce carbon footprint by up to 30%.",
    "Economic analysis indicates that diversification into renewable energy storage solutions could improve mine viability."
]

@app.get("/")
async def root():
    return {
        "message": "FVI Analytics Platform API",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "running",
            "database": "connected",
            "cache": "connected"
        }
    }

@app.get("/metrics")
async def get_metrics():
    return {
        "data": MOCK_METRICS,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/catalog")
async def get_catalog():
    return {
        "total_metrics": 46,
        "thematic_areas": 5,
        "data_sources": 92,
        "last_updated": datetime.now().isoformat(),
        "status": "active"
    }

@app.post("/chat")
async def chat_endpoint(message: dict):
    """Simple chat endpoint that returns mock responses"""
    import random
    
    user_message = message.get("message", "")
    
    # Simple keyword-based responses
    if "coal" in user_message.lower():
        response = "Based on our coal market analysis, the global demand has shifted significantly due to renewable energy adoption."
    elif "fvi" in user_message.lower() or "score" in user_message.lower():
        response = "The FVI scoring model indicates that mines with strong ESG profiles show 23% better long-term viability."
    elif "environment" in user_message.lower():
        response = "Environmental impact assessments show that modern mining techniques can reduce carbon footprint by up to 30%."
    elif "economic" in user_message.lower():
        response = "Economic analysis indicates that diversification into renewable energy storage solutions could improve mine viability."
    else:
        response = random.choice(MOCK_CHAT_RESPONSES)
    
    return {
        "response": response,
        "timestamp": datetime.now().isoformat(),
        "model": "FVI-Analytics-LLM",
        "confidence": 0.92
    }

@app.get("/mine-profiles")
async def get_mine_profiles():
    """Get mock mine profile data"""
    return {
        "data": [
            {"id": "mine_001", "name": "Alpha Coal Mine", "fvi_score": 78.5, "status": "active"},
            {"id": "mine_002", "name": "Beta Mining Co", "fvi_score": 65.2, "status": "active"},
            {"id": "mine_003", "name": "Gamma Resources", "fvi_score": 82.1, "status": "active"},
            {"id": "mine_004", "name": "Delta Energy", "fvi_score": 71.8, "status": "monitoring"},
        ],
        "total": 4,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/system-metrics")
async def get_system_metrics():
    """Get system performance metrics"""
    return {
        "data": {
            "cpu_usage": 45.2,
            "memory_usage": 62.8,
            "api_requests_per_minute": 127,
            "database_connections": 8,
            "cache_hit_rate": 0.94
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/data-quality")
async def get_data_quality():
    """Get data quality metrics"""
    return {
        "data": {
            "completeness": 0.89,
            "accuracy": 0.94,
            "freshness": 0.87,
            "consistency": 0.91
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "simple_api:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
