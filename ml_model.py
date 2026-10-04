import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_and_save_model():
    print("🚀 Starting DMDW ML Pipeline...")
    
    # 1. Load Data
    try:
        df = pd.read_csv('defects_data.csv')
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print("❌ Error: 'defects_data.csv' not found. Please ensure the file is in the same folder.")
        return

    # 2. Data Preprocessing & Feature Engineering
    # Drop unique identifiers that don't help prediction
    df = df.drop(['defect_id', 'product_id'], axis=1)
    
    # Convert date to datetime and extract useful features (Month, Day of Week)
    df['defect_date'] = pd.to_datetime(df['defect_date'])
    df['Month'] = df['defect_date'].dt.month
    df['DayOfWeek'] = df['defect_date'].dt.dayofweek
    df = df.drop('defect_date', axis=1) # Drop original date column

    # Define Features (X) and Target (y)
    # We will predict the 'defect_type' (Structural, Functional, Cosmetic)
    X = df.drop('defect_type', axis=1)
    y = df['defect_type']

    # 3. Encoding Categorical Variables
    # We need to convert text to numbers for the ML model
    encoders = {}
    categorical_cols = ['defect_location', 'severity', 'inspection_method']
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    
    # Encode the Target variable
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    encoders['target'] = target_encoder

    print("✅ Preprocessing and Encoding complete.")

    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # 5. Model Training (Random Forest Classifier)
    print("🧠 Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # 6. Model Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

    # 7. Save Model and Encoders
    joblib.dump(model, 'defect_model.pkl')
    joblib.dump(encoders, 'encoders.pkl')
    print("💾 Model and Encoders saved as 'defect_model.pkl' and 'encoders.pkl'")

if __name__ == "__main__":
    train_and_save_model()