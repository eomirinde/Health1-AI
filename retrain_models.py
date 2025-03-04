import os
import pandas as pd
from src.models.data_preprocessing import load_data, preprocess_symptom_data, preprocess_predictive_data
from src.models.symptom_checker_model import create_model as create_symptom_checker_model
from src.models.predictive_analysis_model import create_model as create_predictive_analysis_model

def retrain_symptom_checker():
    # Check for new data
    data = load_data('data/symptom_data.csv')
    
    # Preprocess data
    X_train, X_test, y_train, y_test, label_encoder = preprocess_symptom_data(data)
    
    # Create and train model
    model = create_symptom_checker_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    
    # Save the trained model and label encoder
    model.save('models/symptom_checker_model.h5')
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

def retrain_predictive_analysis():
    # Check for new data
    data = load_data('data/patient_history.csv')
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_predictive_data(data)
    
    # Create and train model
    model = create_predictive_analysis_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    
    # Save the trained model and scaler
    model.save('models/predictive_analysis_model.h5')
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    retrain_symptom_checker()
    retrain_predictive_analysis()