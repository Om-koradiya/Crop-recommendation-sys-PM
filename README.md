# 🌾 AgriAI: Smart Crop Recommendation System

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)

AgriAI is a machine learning-powered web application that helps farmers and agricultural planners make data-driven decisions. By analyzing soil nutrients (Nitrogen, Phosphorus, Potassium) and environmental factors (Temperature, Humidity, pH, Rainfall), it recommends the most optimal crop to maximize yield.

## ✨ Features

- **Multi-page Streamlit Interface:** Modern, dark-themed, and responsive UI.
- **Data Exploration Dashboard:** View dataset characteristics, feature distributions, and model comparison metrics natively within the app.
- **Accurate Predictions:** Uses an advanced `Random Forest Classifier` model trained on over 2,200 instances with an accuracy of **99.5%**.
- **Probability Scores:** Provides confidence levels for the top recommended crop, along with alternative crop suggestions.
- **One-Click Deployable:** Ready to be hosted directly on Streamlit Community Cloud.

## 📊 Dataset

The model is trained on a comprehensive dataset (`crop_recommendation.csv`) which includes 22 unique crop classes.
* **Features Used:** `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`
* **Target Variable:** `label` (Crop name)

## 🛠️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Om-koradiya/Crop-recommendation-app.git
   cd Crop-recommendation-app
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Re-train the model:**
   If you want to re-train the models and generate new metadata:
   ```bash
   python train.py
   ```

4. **Run the Streamlit Application:**
   ```bash
   streamlit run app.py
   ```

## 🧠 Machine Learning Pipeline

The `train.py` script covers the entire pipeline:
- **Preprocessing:** `StandardScaler` for numerical features and `LabelEncoder` for crop targets.
- **Model Evaluation:** Compares `Random Forest`, `Extra Trees`, and `XGBoost`.
- **Artifacts Generation:** Automatically saves the best model (`best_model.pkl`), scalers (`scaler.pkl`, `label_encoder.pkl`), and metrics (`feature_importance.json`, `model_results.json`) inside the `models/` directory for the Streamlit app to consume.

## 🚀 Deployment

This project is fully ready for deployment on **Streamlit Community Cloud**:
1. Push this repository to GitHub.
2. Log in to [share.streamlit.io](https://share.streamlit.io).
3. Create a new app, select this repository, and set the main file path to `app.py`.
4. Deploy!
