from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import logging
from datetime import datetime
from ml_models import MFIModelTrainer



app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the ML model
try:
    predictor = MFIModelTrainer()
    logger.info("ML Model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ML model: {str(e)}")
    predictor = None

# Simple counter for statistics
prediction_count = 0

@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle loan prediction requests"""
    global prediction_count
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['age', 'gender', 'income', 'loanAmount', 'education', 
                          'employment', 'maritalStatus', 'location', 'mobileUsage', 
                          'transactionFreq', 'previousLoans', 'previousDefaults']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Validate data types and ranges
        try:
            data['age'] = int(data['age'])
            data['income'] = float(data['income'])
            data['loanAmount'] = float(data['loanAmount'])
            data['mobileUsage'] = int(data['mobileUsage'])
            data['transactionFreq'] = int(data['transactionFreq'])
            data['previousLoans'] = int(data['previousLoans'])
            data['previousDefaults'] = int(data['previousDefaults'])
        except ValueError:
            return jsonify({'error': 'Invalid data types in request'}), 400
        
        # Business rule validations
        if not (18 <= data['age'] <= 80):
            return jsonify({'error': 'Age must be between 18 and 80'}), 400
        
        if not (5 <= data['loanAmount'] <= 12):
            return jsonify({'error': 'Loan amount must be between 5 and 12 IDR'}), 400
        
        if data['previousDefaults'] > data['previousLoans']:
            return jsonify({'error': 'Previous defaults cannot exceed previous loans'}), 400
        
        if data['income'] <= 0:
            return jsonify({'error': 'Income must be positive'}), 400
        
        # Make prediction using the ML model
        if predictor is None:
            return jsonify({'error': 'ML model not available'}), 500
        
        result = predictor.predict(data)
        
        # Increment counter
        prediction_count += 1
        
        # Log prediction (in production, you might want to store this in a database)
        logger.info(f"Prediction #{prediction_count} completed for customer age {data['age']}, risk: {result['riskPercentage']}%")
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat(),
            'predictionId': prediction_count
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/stats')
def get_stats():
    """Get application statistics"""
    return jsonify({
        'total_predictions': prediction_count,
        'model_version': '1.0.0',
        'model_accuracy': '87%',
        'uptime': 'Available',
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)