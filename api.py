#!/usr/bin/env python3
"""
Version: 1.0.0
ImmunePlan REST API
Multi-doctor LLM API with support for different specialist types
Currently implemented: Dr. Immunity (Autoimmune Specialist)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import requests

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Doctor types configuration
DOCTOR_TYPES = {
    'immune': {
        'name': 'Dr. Immunity',
        'specialty': 'Autoimmune Specialist',
        'expertise': 'IBD, RA, and immune system disorders',
        'available': True,
        'model_endpoint': os.getenv('IMMUNE_LLM_ENDPOINT', 'http://localhost:8000/api/generate'),
        'model_type': 'immune_specialist'
    },
    'onco': {
        'name': 'Dr. Onco',
        'specialty': 'Oncology Specialist',
        'expertise': 'Comprehensive cancer care and screening guidance',
        'available': False,  # Not yet implemented
        'model_endpoint': None,
        'model_type': 'oncology_specialist'
    },
    'cardio': {
        'name': 'Dr. Cardio',
        'specialty': 'Heart Disease Specialist',
        'expertise': 'Cardiovascular health, hypertension, and heart disease management',
        'available': False,  # Not yet implemented
        'model_endpoint': None,
        'model_type': 'cardiology_specialist'
    },
    'mind': {
        'name': 'Dr. Mind',
        'specialty': 'Mental Health Specialist',
        'expertise': 'Support for anxiety, depression, and mental well-being',
        'available': False,  # Not yet implemented
        'model_endpoint': None,
        'model_type': 'mental_health_specialist'
    },
    'fit': {
        'name': 'Coach Fit',
        'specialty': 'Fitness & Lifestyle',
        'expertise': 'Proactive health, nutrition, and exercise planning',
        'available': False,  # Not yet implemented
        'model_endpoint': None,
        'model_type': 'fitness_coach'
    },
    'nature': {
        'name': 'Dr. Nature',
        'specialty': 'Naturopathic Medicine',
        'expertise': 'Holistic approaches and natural remedies',
        'available': False,  # Not yet implemented
        'model_endpoint': None,
        'model_type': 'naturopathic_specialist'
    }
}

def call_immune_llm(message, context=None, history=None):
    """
    Call the Immune Specialist LLM
    Connects to the trained medical LLM for autoimmune conditions
    """
    # Get LLM endpoint from environment or use default
    llm_endpoint = os.getenv('IMMUNE_LLM_ENDPOINT', 'http://localhost:8000/generate')
    
    try:
        # Prepare request payload
        payload = {
            'message': message,
            'context': context or {},
            'history': history or [],
            'doctor_type': 'immune',
            'specialty': 'Autoimmune Specialist'
        }
        
        # Call LLM API
        response = requests.post(
            llm_endpoint,
            json=payload,
            timeout=30,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            llm_data = response.json()
            
            # Extract response from LLM
            # Adjust these keys based on your LLM API response format
            llm_response = llm_data.get('response') or llm_data.get('text') or llm_data.get('output', '')
            
            return {
                'response': llm_response,
                'disclaimer': 'This is for informational support only. Always consult with qualified healthcare professionals for medical advice and treatment decisions.',
                'doctor_type': 'immune',
                'doctor_name': 'Dr. Immunity',
                'llm_metadata': llm_data.get('metadata', {})
            }
        else:
            logger.error(f"LLM API returned status {response.status_code}: {response.text}")
            # Fallback response
            return {
                'response': f"I'm Dr. Immunity, your Autoimmune Specialist. I'm having trouble processing your question right now. Please try rephrasing your question or contact your healthcare provider for immediate assistance.",
                'disclaimer': 'This is for informational support only. Always consult with qualified healthcare professionals for medical advice and treatment decisions.',
                'doctor_type': 'immune',
                'doctor_name': 'Dr. Immunity'
            }
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling LLM endpoint: {e}")
        # Fallback response if LLM is unavailable
        return {
            'response': f"I'm Dr. Immunity, your Autoimmune Specialist. I understand you're asking about: {message}. While I can provide educational information about autoimmune conditions, IBD, and RA, please remember this is for informational support only. Always consult with your healthcare provider for medical advice.",
            'disclaimer': 'This is for informational support only. Always consult with qualified healthcare professionals for medical advice and treatment decisions.',
            'doctor_type': 'immune',
            'doctor_name': 'Dr. Immunity',
            'note': 'LLM service temporarily unavailable, using fallback response'
        }
    except Exception as e:
        logger.error(f"Unexpected error in call_immune_llm: {e}", exc_info=True)
        # Fallback response
        return {
            'response': "I apologize, but I'm experiencing technical difficulties. Please try again later or consult with your healthcare provider for immediate assistance.",
            'disclaimer': 'This is for informational support only. Always consult with qualified healthcare professionals for medical advice and treatment decisions.',
            'doctor_type': 'immune',
            'doctor_name': 'Dr. Immunity'
        }

@app.route('/api/doctors', methods=['GET'])
def get_doctors():
    """Get list of available doctors"""
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
    """Main chat endpoint - routes to appropriate doctor LLM"""
    try:
        data = request.json
        
        # Validate required fields
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        message = data.get('message', '').strip()
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Get doctor type (default to immune if not specified)
        doctor_type = data.get('doctorType', 'immune').lower()
        
        # Validate doctor type
        if doctor_type not in DOCTOR_TYPES:
            return jsonify({
                'error': f'Invalid doctor type. Available types: {", ".join(DOCTOR_TYPES.keys())}'
            }), 400
        
        doctor_info = DOCTOR_TYPES[doctor_type]
        
        # Check if doctor is available
        if not doctor_info['available']:
            return jsonify({
                'error': f'{doctor_info["name"]} is not yet available. Currently only Dr. Immunity is available.',
                'available_doctors': [d for d in DOCTOR_TYPES.values() if d['available']]
            }), 503
        
        # Get context if provided
        context = data.get('context', {})
        conversation_history = data.get('history', [])
        
        # Route to appropriate LLM based on doctor type
        if doctor_type == 'immune':
            result = call_immune_llm(message, context, conversation_history)
        else:
            # Placeholder for other doctor types
            return jsonify({
                'error': f'{doctor_info["name"]} is not yet implemented',
                'available_doctors': [d for d in DOCTOR_TYPES.values() if d['available']]
            }), 503
        
        # Add metadata
        result['doctor_type'] = doctor_type
        result['doctor_name'] = doctor_info['name']
        result['timestamp'] = os.popen('date -u +"%Y-%m-%dT%H:%M:%SZ"').read().strip()
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

@app.route('/api/doctor/<doctor_type>', methods=['GET'])
def get_doctor_info(doctor_type):
    """Get information about a specific doctor"""
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

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'ImmunePlan API',
        'available_doctors': len([d for d in DOCTOR_TYPES.values() if d['available']]),
        'total_doctors': len(DOCTOR_TYPES)
    }), 200

@app.route('/', methods=['GET'])
def index():
    """API information"""
    return jsonify({
        'service': 'ImmunePlan REST API',
        'version': '1.0.0',
        'endpoints': {
            'GET /api/doctors': 'List all available doctors',
            'GET /api/doctor/<type>': 'Get specific doctor information',
            'POST /api/chat': 'Chat with a doctor (requires doctorType and message)',
            'GET /health': 'Health check'
        },
        'available_doctors': [d['name'] for d in DOCTOR_TYPES.values() if d['available']]
    }), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting ImmunePlan API on {host}:{port}")
    logger.info(f"Available doctors: {[d['name'] for d in DOCTOR_TYPES.values() if d['available']]}")
    
    app.run(host=host, port=port, debug=debug)

