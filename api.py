#!/usr/bin/env python3
"""
ImmunePlan REST API v2.0
Full reliability pipeline — structured prompts, RAG, tools, agents, safety, monitoring.
"""
import sys
import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import asyncio

app = FastAPI(title="ImmunePlan REST API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOCTOR_TYPES = {
    'immune': {
        'name': 'Dr. Immunity',
        'specialty': 'Autoimmune Disease Specialist',
        'expertise': 'Rheumatoid arthritis, Crohn\'s disease/IBD, lupus (SLE), Sjogren\'s syndrome, and related autoimmune conditions',
        'available': True
    },
}

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from pipeline.lib.tools import create_medical_tool_registry
        from pipeline.lib.rag import VectorStore, RAGPipeline
        from pipeline.lib.safety import SafetyPipeline
        from pipeline.lib.monitoring import MonitoringService
        from pipeline.lib.agent import MedicalAgent, MultiAgentOrchestrator
        from pipeline.config import DATA_DIR

        monitoring = MonitoringService()
        tools = create_medical_tool_registry()
        store = VectorStore()
        vs_path = DATA_DIR / "vector_store"
        if vs_path.exists():
            store.load(str(vs_path))
        rag = RAGPipeline(store)
        safety = SafetyPipeline()
        agent = MedicalAgent(tool_registry=tools, rag_pipeline=rag,
                             safety_pipeline=safety, monitoring=monitoring)
        orchestrator = MultiAgentOrchestrator(monitoring)
        orchestrator.register_agent("immune", agent)
        _pipeline = {"orchestrator": orchestrator, "monitoring": monitoring,
                      "tools": tools, "safety": safety}
    return _pipeline

class ChatRequest(BaseModel):
    message: str
    doctorType: str = "immune"

class ToolRequest(BaseModel):
    # Dynamic fields for tool arguments
    pass

class FeedbackRequest(BaseModel):
    request_id: str
    rating: int
    feedback: str = ""

@app.post("/api/chat")
async def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    doctor_type = request.doctorType.lower()
    if doctor_type not in DOCTOR_TYPES:
        raise HTTPException(status_code=400, detail="Invalid doctor type")
    if not DOCTOR_TYPES[doctor_type]['available']:
        raise HTTPException(status_code=503, detail=f"{DOCTOR_TYPES[doctor_type]['name']} not available yet")

    p = get_pipeline()
    
    # Run the blocking agent execution in a thread pool so we don't block the async event loop
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, p["orchestrator"].route_and_process, message, doctor_type)
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return {
        'response': result.response,
        'doctor_name': DOCTOR_TYPES[doctor_type]['name'],
        'doctor_type': doctor_type,
        'intent': result.intent.value,
        'confidence': result.confidence,
        'quality_score': result.quality_score,
        'citations': result.citations,
        'steps': [{"type": s.step_type, "content": s.content} for s in result.steps],
        'processing_time': result.processing_time_seconds,
        'disclaimer': result.disclaimer,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }

@app.get("/api/doctors")
async def get_doctors():
    return {'doctors': [
        {'id': k, **{dk: dv for dk, dv in v.items()}} for k, v in DOCTOR_TYPES.items()
    ]}

@app.get("/api/tools")
async def list_tools():
    return {'tools': get_pipeline()["tools"].list_tools()}

@app.post("/api/tools/{tool_name}")
async def execute_tool(tool_name: str, request: Request):
    args = await request.json() if request.headers.get('content-type') == 'application/json' else {}
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: get_pipeline()["tools"].execute(tool_name, **args))
    return result.to_dict()

@app.get("/api/dashboard")
async def dashboard():
    return get_pipeline()["monitoring"].get_dashboard()

@app.post("/api/feedback")
async def submit_feedback(data: FeedbackRequest):
    get_pipeline()["monitoring"].record_feedback(data.request_id, data.rating, data.feedback)
    return {'status': 'recorded'}

@app.get("/api/reliability")
async def reliability_spec():
    from pipeline.config import RELIABILITY_SPEC
    return RELIABILITY_SPEC

@app.get("/health")
async def health():
    return {'status': 'ok', 'service': 'ImmunePlan API v2.0'}

@app.get("/")
async def index():
    return {
        'service': 'ImmunePlan REST API', 'version': '2.0.0',
        'endpoints': {
            'POST /api/chat': 'Chat with reliability pipeline',
            'GET /api/doctors': 'List doctors',
            'GET /api/tools': 'List medical tools',
            'POST /api/tools/<name>': 'Execute tool',
            'GET /api/dashboard': 'Monitoring dashboard',
            'POST /api/feedback': 'Submit feedback',
            'GET /api/reliability': 'Reliability spec',
        },
    }

if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    uvicorn.run("api:app", host=host, port=port, reload=debug)