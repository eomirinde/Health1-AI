from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pickle

app = Flask(__name__)

# Load models and encoders
symptom_checker_model = load_model('models/symptom_checker_model.h5')
with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

predictive_analysis_model = load_model('models/predictive_analysis_model.h5')
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/predict-symptom', methods=['POST'])
def predict_symptom():
    data = request.json
    symptoms = np.array([data['symptoms']])
    prediction = symptom_checker_model.predict(symptoms)
    diagnosis = label_encoder.inverse_transform([np.argmax(prediction)])
    return jsonify({'diagnosis': diagnosis[0]})

@app.route('/predict-analysis', methods=['POST'])
def predict_analysis():
    data = request.json
    patient_history = np.array([data['patient_history']])
    patient_history = scaler.transform(patient_history)
    prediction = predictive_analysis_model.predict(patient_history)
    return jsonify({'risk': prediction[0][0]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')