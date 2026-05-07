import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Ensure directories exist
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 1. Load Data
df = pd.read_csv('dataset/crop_recommendation.csv')

# Use only relevant features for Crop Recommendation
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
target = 'label'

X = df[features]
y = df[target]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 2. Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# 4. Define Models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

# 5. Train and Evaluate Models
results = {}
best_model = None
best_accuracy = 0
best_name = ""

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results[name] = {'Accuracy': round(acc, 4), 'F1-Score': round(f1, 4)}
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_name = name

print("\nModel Comparison:")
for name, metrics in results.items():
    print(f"{name}: Accuracy = {metrics['Accuracy']}, F1-Score = {metrics['F1-Score']}")

print(f"\nBest Model: {best_name} with Accuracy: {best_accuracy}")

# 6. Save Best Model and Metadata
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

metadata = {
    'best_model': best_name,
    'accuracy': best_accuracy,
    'features': features,
    'crop_classes': list(le.classes_)
}

with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f)

# 7. Extract Feature Importances
try:
    importances = best_model.feature_importances_
    importance_dict = {name: float(imp) for name, imp in zip(features, importances)}
    
    with open('models/feature_importance.json', 'w') as f:
        json.dump(importance_dict, f)
    print("Feature importance saved.")
except Exception as e:
    print(f"Could not extract feature importance: {e}")

# Save results for Streamlit
with open('models/model_results.json', 'w') as f:
    json.dump(results, f)

print("Training Complete!")
