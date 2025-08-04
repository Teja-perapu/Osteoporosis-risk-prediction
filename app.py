from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Path to model and preprocessor
MODEL_PATH = "backend/Output/osteoporosis_lightgbm_model.pkl"
PREPROCESSOR_PATH = "backend/Output/osteoporosis_preprocessor.pkl"

def predict_osteoporosis_risk(patient_data):
    """
    Make osteoporosis risk prediction based on patient data
    """
    try:
        # Load model and preprocessor
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        
        # Convert the data to a pandas DataFrame
        df = pd.DataFrame([patient_data])
        
        # Rename columns to match training data if needed
        column_mapping = {
            'hormonalChanges': 'Hormonal Changes',
            'familyHistory': 'Family History',
            'ethnicity': 'Race/Ethnicity',
            'bodyWeight': 'Body Weight',
            'calciumIntake': 'Calcium Intake',
            'vitaminDIntake': 'Vitamin D Intake',
            'physicalActivity': 'Physical Activity',
            'smoking': 'Smoking',
            'alcoholConsumption': 'Alcohol Consumption',
            'medicalConditions': 'Medical Conditions',
            'medications': 'Medications',
            'priorFractures': 'Prior Fractures',
            'gender': 'Gender',
            'age': 'Age'
        }
        df = df.rename(columns=column_mapping)
        
        # Preprocess the data
        processed_data = preprocessor.transform(df)
        
        # Make prediction
        prediction = int(model.predict(processed_data)[0])
        probability = float(model.predict_proba(processed_data)[0, 1])
        
        # Determine risk level
        risk_level = 'High' if probability >= 0.75 else ('Moderate' if probability >= 0.5 else 'Low')
        
        # Prepare confidence score (adjusted probability as percentage)
        confidence = min(round(probability * 100), 100) if prediction == 1 else min(round((1-probability) * 100), 100)
        
        # Generate recommendation message
        if risk_level == 'High':
            message = "We recommend consulting with a healthcare provider for a bone density scan and comprehensive assessment."
        elif risk_level == 'Moderate':
            message = "Consider discussing risk factors with your doctor during your next visit and maintain bone-healthy habits."
        else:
            message = "Your risk appears low. Continue maintaining bone health through proper nutrition and exercise."
            
        return {
            "risk": risk_level,
            "confidence": confidence,
            "message": message
        }
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {"error": str(e)}

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for osteoporosis risk prediction
    """
    try:
        data = request.json
        result = predict_osteoporosis_risk(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({"status": "healthy", "message": "The API is running"})

if __name__ == '__main__':
    # Check if model and preprocessor exist
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        print(f"Warning: Model or preprocessor files not found. Please ensure they exist at {MODEL_PATH} and {PREPROCESSOR_PATH}")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)