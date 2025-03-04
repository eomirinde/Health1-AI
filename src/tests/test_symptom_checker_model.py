import pytest
import numpy as np
from tensorflow.keras.models import load_model
import pickle

@pytest.fixture
def symptom_checker_model():
    model = load_model('models/symptom_checker_model.h5')
    with open('models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

def test_symptom_checker_model(symptom_checker_model):
    model, label_encoder = symptom_checker_model
    symptoms = np.array([[1, 0, 0, 1, 0, 1, 1]])
    prediction = model.predict(symptoms)
    diagnosis = label_encoder.inverse_transform([np.argmax(prediction)])
    assert diagnosis is not None