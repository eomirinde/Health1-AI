import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_symptom_data(data):
    # Encode categorical labels
    label_encoder = LabelEncoder()
    data['Diagnosis'] = label_encoder.fit_transform(data['Diagnosis'])

    # Split data into features and labels
    X = data.drop('Diagnosis', axis=1)
    y = data['Diagnosis']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder

def preprocess_predictive_data(data):
    # Separate features and labels
    X = data.drop('Target', axis=1)
    y = data['Target']

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler