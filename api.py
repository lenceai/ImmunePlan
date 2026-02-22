#!/usr/bin/env python3
"""
Version: 2.0.0
ImmunePlan REST API with Full Reliability Framework

Multi-doctor LLM API implementing all reliability concepts:
  - Layer 1: Reliable Outputs (structured prompts, RAG grounding)
  - Layer 2: Reliable Agents (tools, memory, intent routing)
  - Layer 3: Reliable Operations (monitoring, safety, evaluation)

Currently implemented: Dr. Immunity (Autoimmune Specialist)
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
    'immune': {
        'name': 'Dr. Immunity',
        'specialty': 'Autoimmune Specialist',
        'expertise': 'IBD, RA, and immune system disorders',
        'available': True,
        'model_type': 'immune_specialist'
    },
    'onco': {
        'name': 'Dr. Onco',
        'specialty': 'Oncology Specialist',
        'expertise': 'Comprehensive cancer care and screening guidance',
        'available': False,
        'model_type': 'oncology_specialist'
    },
    'cardio': {
        'name': 'Dr. Cardio',
        'specialty': 'Heart Disease Specialist',
        'expertise': 'Cardiovascular health, hypertension, and heart disease management',
        'available': False,
        'model_type': 'cardiology_specialist'
    },
    'mind': {
        'name': 'Dr. Mind',
        'specialty': 'Mental Health Specialist',
        'expertise': 'Support for anxiety, depression, and mental well-being',
        'available': False,
        'model_type': 'mental_health_specialist'
    },
    'fit': {
        'name': 'Coach Fit',
        'specialty': 'Fitness & Lifestyle',
        'expertise': 'Proactive health, nutrition, and exercise planning',
        'available': False,
        'model_type': 'fitness_coach'
    },
    'nature': {
        'name': 'Dr. Nature',
        'specialty': 'Naturopathic Medicine',
        'expertise': 'Holistic approaches and natural remedies',
        'available': False,
        'model_type': 'naturopathic_specialist'
    }
}


def get_reliability_pipeline():
    """Lazy-initialize the reliability pipeline."""
    from reliability.pipeline import get_pipeline
    pipeline = get_pipeline()
    pipeline.initialize()
    return pipeline


@app.route('/api/doctors', methods=['GET'])
def get_doctors():
    """Get list of available doctors."""
    doctors = []
    for doctor_id, doctor_info in DOCTOR_TYPES.items():
        doctors.append({
            'id': doctor_id,
            'name': doctor_info['name'],
            'specialty': doctor_info['specialty'],
            'expertise': doctor_info['expertise'],
            'available': doctor_info['available']
        })
    return jsonify({'doctors': doctors}), 200


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint with full reliability pipeline.

    Routes through:
    1. Input safety check (PII detection, content filter)
    2. Intent classification and agent routing
    3. RAG retrieval for context grounding
    4. Tool usage (lab lookup, drug info, guidelines)
    5. Response generation with structured prompts
    6. Output quality evaluation (groundedness, structure)
    7. Output safety check and sanitization
    8. Monitoring and logging
    """
    try:
        data = request.json

        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400

        message = data.get('message', '').strip()
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400

        doctor_type = data.get('doctorType', 'immune').lower()

        if doctor_type not in DOCTOR_TYPES:
            return jsonify({
                'error': f'Invalid doctor type. Available: {", ".join(DOCTOR_TYPES.keys())}'
            }), 400

        doctor_info = DOCTOR_TYPES[doctor_type]

        if not doctor_info['available']:
            return jsonify({
                'error': f'{doctor_info["name"]} is not yet available.',
                'available_doctors': [
                    {'id': k, 'name': v['name']}
                    for k, v in DOCTOR_TYPES.items() if v['available']
                ]
            }), 503

        context = data.get('context', {})
        pipeline = get_reliability_pipeline()

        result = pipeline.process_query(
            query=message,
            doctor_type=doctor_type,
            context=context,
        )

        result['doctor_type'] = doctor_type
        result['doctor_name'] = doctor_info['name']
        result['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500


@app.route('/api/doctor/<doctor_type>', methods=['GET'])
def get_doctor_info(doctor_type):
    """Get information about a specific doctor."""
    doctor_type = doctor_type.lower()

    if doctor_type not in DOCTOR_TYPES:
        return jsonify({
            'error': f'Doctor type "{doctor_type}" not found',
            'available_types': list(DOCTOR_TYPES.keys())
        }), 404

    doctor_info = DOCTOR_TYPES[doctor_type]
    return jsonify({
        'id': doctor_type,
        'name': doctor_info['name'],
        'specialty': doctor_info['specialty'],
        'expertise': doctor_info['expertise'],
        'available': doctor_info['available']
    }), 200


@app.route('/api/tools', methods=['GET'])
def list_tools():
    """List all available medical tools and their schemas."""
    pipeline = get_reliability_pipeline()
    return jsonify({'tools': pipeline.get_tool_schemas()}), 200


@app.route('/api/tools/<tool_name>', methods=['POST'])
def execute_tool(tool_name):
    """Execute a specific medical tool directly."""
    try:
        data = request.json or {}
        pipeline = get_reliability_pipeline()
        result = pipeline.tool_registry.execute(tool_name, **data)
        return jsonify(result.to_dict()), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dashboard', methods=['GET'])
def dashboard():
    """Get monitoring dashboard with system health and quality metrics."""
    pipeline = get_reliability_pipeline()
    return jsonify(pipeline.get_dashboard()), 200


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback for a request."""
    data = request.json
    if not data or 'request_id' not in data or 'rating' not in data:
        return jsonify({'error': 'request_id and rating are required'}), 400

    pipeline = get_reliability_pipeline()
    pipeline.submit_feedback(
        request_id=data['request_id'],
        rating=data['rating'],
        feedback=data.get('feedback', ''),
    )
    return jsonify({'status': 'feedback_recorded'}), 200


@app.route('/api/reliability', methods=['GET'])
def reliability_spec():
    """Get the reliability specification for the system."""
    from reliability.config import RELIABILITY_SPEC
    return jsonify({
        'project': RELIABILITY_SPEC.project_name,
        'tier': RELIABILITY_SPEC.tier.value,
        'reliability_definition': RELIABILITY_SPEC.reliability_definition,
        'quality_targets': RELIABILITY_SPEC.quality_targets,
        'latency_targets': RELIABILITY_SPEC.latency_targets,
        'allowed_actions': RELIABILITY_SPEC.allowed_actions,
        'forbidden_actions': RELIABILITY_SPEC.forbidden_actions,
    }), 200


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    pipeline = get_reliability_pipeline()
    dashboard_data = pipeline.get_dashboard()

    return jsonify({
        'status': dashboard_data.get('status', 'ok'),
        'service': 'ImmunePlan API v2.0 (Reliability Framework)',
        'available_doctors': len([d for d in DOCTOR_TYPES.values() if d['available']]),
        'total_doctors': len(DOCTOR_TYPES),
        'total_requests': dashboard_data.get('total_requests', 0),
        'reliability_tier': 'clinical_support',
    }), 200


@app.route('/', methods=['GET'])
def index():
    """API information."""
    return jsonify({
        'service': 'ImmunePlan REST API',
        'version': '2.0.0',
        'reliability_framework': 'Building Reliable AI Systems',
        'layers': {
            'layer_1_reliable_outputs': ['structured_prompts', 'rag_grounding', 'fine_tuning'],
            'layer_2_reliable_agents': ['tool_integration', 'memory', 'intent_routing'],
            'layer_3_reliable_operations': ['evaluation', 'monitoring', 'safety', 'responsible_ai'],
        },
        'endpoints': {
            'GET /': 'API information',
            'GET /health': 'Health check with reliability status',
            'GET /api/doctors': 'List all available doctors',
            'GET /api/doctor/<type>': 'Get specific doctor information',
            'POST /api/chat': 'Chat with a doctor (full reliability pipeline)',
            'GET /api/tools': 'List available medical tools',
            'POST /api/tools/<name>': 'Execute a medical tool directly',
            'GET /api/dashboard': 'Monitoring dashboard',
            'POST /api/feedback': 'Submit user feedback',
            'GET /api/reliability': 'System reliability specification',
        },
        'available_doctors': [d['name'] for d in DOCTOR_TYPES.values() if d['available']]
    }), 200


if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('DEBUG', 'False').lower() == 'true'

    logger.info(f"Starting ImmunePlan API v2.0 on {host}:{port}")
    logger.info(f"Reliability Framework: Active")
    logger.info(f"Available doctors: {[d['name'] for d in DOCTOR_TYPES.values() if d['available']]}")

    app.run(host=host, port=port, debug=debug)
