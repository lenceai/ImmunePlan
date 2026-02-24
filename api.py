#!/usr/bin/env python3
"""
ImmunePlan REST API v2.0
Full reliability pipeline — structured prompts, RAG, tools, agents, safety, monitoring.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import time

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOCTOR_TYPES = {
    'immune': {'name': 'Dr. Immunity', 'specialty': 'Autoimmune Specialist',
               'expertise': 'IBD, RA, and immune system disorders', 'available': True},
    'onco':   {'name': 'Dr. Onco', 'specialty': 'Oncology Specialist',
               'expertise': 'Cancer care and screening', 'available': False},
    'cardio': {'name': 'Dr. Cardio', 'specialty': 'Heart Disease Specialist',
               'expertise': 'Cardiovascular health', 'available': False},
    'mind':   {'name': 'Dr. Mind', 'specialty': 'Mental Health Specialist',
               'expertise': 'Anxiety, depression, mental well-being', 'available': False},
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


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        message = data['message'].strip()
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400

        doctor_type = data.get('doctorType', 'immune').lower()
        if doctor_type not in DOCTOR_TYPES:
            return jsonify({'error': f'Invalid doctor type'}), 400
        if not DOCTOR_TYPES[doctor_type]['available']:
            return jsonify({'error': f'{DOCTOR_TYPES[doctor_type]["name"]} not available yet'}), 503

        p = get_pipeline()
        result = p["orchestrator"].route_and_process(message, doctor_type)

        return jsonify({
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
        }), 200
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/doctors', methods=['GET'])
def get_doctors():
    return jsonify({'doctors': [
        {'id': k, **{dk: dv for dk, dv in v.items()}} for k, v in DOCTOR_TYPES.items()
    ]}), 200


@app.route('/api/tools', methods=['GET'])
def list_tools():
    return jsonify({'tools': get_pipeline()["tools"].list_tools()}), 200


@app.route('/api/tools/<tool_name>', methods=['POST'])
def execute_tool(tool_name):
    result = get_pipeline()["tools"].execute(tool_name, **(request.json or {}))
    return jsonify(result.to_dict()), 200


@app.route('/api/dashboard', methods=['GET'])
def dashboard():
    return jsonify(get_pipeline()["monitoring"].get_dashboard()), 200


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    if not data or 'request_id' not in data or 'rating' not in data:
        return jsonify({'error': 'request_id and rating required'}), 400
    get_pipeline()["monitoring"].record_feedback(data['request_id'], data['rating'], data.get('feedback', ''))
    return jsonify({'status': 'recorded'}), 200


@app.route('/api/reliability', methods=['GET'])
def reliability_spec():
    from pipeline.config import RELIABILITY_SPEC
    return jsonify(RELIABILITY_SPEC), 200


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'ImmunePlan API v2.0'}), 200


@app.route('/', methods=['GET'])
def index():
    return jsonify({
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
    }), 200


if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    host = os.getenv('HOST', '0.0.0.0')
    app.run(host=host, port=port, debug=os.getenv('DEBUG', 'false').lower() == 'true')
