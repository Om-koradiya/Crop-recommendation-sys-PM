import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR DARK MODERN UI ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00E676;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161A25;
        border-right: 1px solid #2B303D;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #00E676;
    }
    
    /* Success Message Box */
    .success-box {
        padding: 20px;
        background-color: rgba(0, 230, 118, 0.1);
        border-left: 5px solid #00E676;
        border-radius: 5px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    /* Prediction Text */
    .pred-text {
        font-size: 32px;
        font-weight: 800;
        color: #00E676;
        margin: 0;
    }
    
    /* Confidence Text */
    .conf-text {
        font-size: 18px;
        color: #A0AEC0;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- CACHE DATA LOADING ---
@st.cache_data
def load_data():
    df = pd.read_csv('dataset/crop_recommendation.csv')
    return df

@st.cache_resource
def load_models():
    model = joblib.load('models/best_model.pkl')
    le = joblib.load('models/label_encoder.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, le, scaler

@st.cache_data
def load_metadata():
    with open('models/metadata.json', 'r') as f:
        meta = json.load(f)
    with open('models/feature_importance.json', 'r') as f:
        feat_imp = json.load(f)
    with open('models/model_results.json', 'r') as f:
        results = json.load(f)
    return meta, feat_imp, results

try:
    df = load_data()
    model, le, scaler = load_models()
    meta, feat_imp, results = load_metadata()
except Exception as e:
    st.error(f"Error loading files. Make sure train.py has been run. Details: {e}")
    st.stop()


# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "",
    ["Project Overview", "Data Exploration", "Crop Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "A machine learning project predicting optimal crops based on soil and climate conditions."
)

# --- PAGE 1: PROJECT OVERVIEW ---
if page == "Project Overview":
    st.title("Crop Recommendation System")
    st.markdown("### A machine learning approach to crop selection")
    
    st.markdown("""
    This project focuses on predicting the most suitable crop to plant based on soil and weather conditions. By analyzing historical data, the model can help maximize crop yield and reduce the chances of failure.
    
    #### The Objective
    The main goal is to recommend the best crop based on the soil's nutrient makeup and the surrounding environment. 
    
    #### The Dataset
    The dataset contains 2,200 rows with the following features:
    * **Nitrogen (N)**: Nitrogen content ratio in the soil
    * **Phosphorus (P)**: Phosphorus content ratio in the soil
    * **Potassium (K)**: Potassium content ratio in the soil
    * **Temperature**: Average temperature in Celsius
    * **Humidity**: Relative humidity percentage
    * **pH Value**: Soil pH level
    * **Rainfall**: Rainfall in mm
    
    #### Methodology
    1. **Data Prep**: I used a `StandardScaler` to normalize the numerical features so they are evaluated evenly.
    2. **Model Training**: I tested a few algorithms like Random Forest, Extra Trees, and XGBoost.
    3. **Selection**: Random Forest was chosen as the final model because it gave the highest accuracy and F1-score without overfitting.
    
    ---
    """)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Size", f"{df.shape[0]:,} rows")
    col2.metric("Total Features", f"{df.shape[1] - 1}")
    col3.metric("Supported Crops", f"{len(meta['crop_classes'])}")


# --- PAGE 2: DATA EXPLORATION ---
elif page == "Data Exploration":
    st.title("Data Exploration")
    st.markdown("Looking at the dataset characteristics and the performance of the models.")
    
    # Dataset Preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("---")
    
    # Crop Profiles
    st.subheader("Ideal Crop Conditions")
    st.markdown("Select a crop to see its average required soil and environmental conditions.")
    selected_crop_profile = st.selectbox("Select Crop:", sorted(df['label'].unique().tolist()))
    
    crop_data = df[df['label'] == selected_crop_profile].mean(numeric_only=True)
    
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Avg Temperature", f"{crop_data['temperature']:.1f} °C")
    col_b.metric("Avg Humidity", f"{crop_data['humidity']:.1f} %")
    col_c.metric("Avg Rainfall", f"{crop_data['rainfall']:.1f} mm")
    col_d.metric("Avg pH", f"{crop_data['ph']:.1f}")
    
    col_e, col_f, col_g = st.columns(3)
    col_e.metric("Nitrogen (N)", f"{crop_data['N']:.1f}")
    col_f.metric("Phosphorus (P)", f"{crop_data['P']:.1f}")
    col_g.metric("Potassium (K)", f"{crop_data['K']:.1f}")
    
    st.markdown("---")
    
    # Feature Distributions
    st.subheader("Feature Distributions")
    selected_feature = st.selectbox("Select Feature to view distribution:", meta['features'])
    
    fig_hist = px.histogram(df, x=selected_feature, color='label', 
                            title=f"Distribution of {selected_feature} across Crops",
                            template="plotly_dark",
                            color_discrete_sequence=px.colors.qualitative.Prism)
    st.plotly_chart(fig_hist, use_container_width=True)
    st.info("Notice how different crops cluster around specific ranges. For example, some crops need a very narrow temperature range, while others are more flexible.")
    
    st.markdown("---")
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    st.markdown("Checking how different features relate to each other.")
    corr = df[meta['features']].corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", 
                         color_continuous_scale='RdBu_r', 
                         title="Feature Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)
    st.info("You can see a strong positive correlation (0.73) between Phosphorus and Potassium, meaning they often appear together in the soil. The weather features (temperature, rainfall) don't really correlate with each other.")
    
    st.markdown("---")
    
    # Boxplot
    st.subheader("Crop Requirement Ranges")
    st.markdown("A closer look at the specific range of a feature required for each crop type.")
    box_feature = st.selectbox("Select Feature for Boxplot:", meta['features'], index=3)
    fig_box = px.box(df, x='label', y=box_feature, color='label',
                     title=f"Required {box_feature} ranges per Crop",
                     template="plotly_dark")
    fig_box.update_layout(xaxis={'categoryorder':'total descending'}, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)
    st.info("This shows the 'sweet spot' for a crop. If the soil feature is outside this box's range, the crop probably won't do well.")

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Comparison")
        model_names = list(results.keys())
        accuracies = [results[m]['Accuracy'] for m in model_names]
        
        fig_models = px.bar(
            x=model_names, y=accuracies, 
            labels={'x': 'Model', 'y': 'Accuracy'},
            title="Accuracy of Evaluated Models",
            template="plotly_dark",
            color=accuracies,
            color_continuous_scale="Viridis"
        )
        fig_models.update_layout(yaxis=dict(range=[0.9, 1.0]))
        st.plotly_chart(fig_models, use_container_width=True)
        st.info("Random Forest and Extra Trees performed the best. Tree-based models are generally great for tabular data like this.")
        
    with col2:
        st.subheader("Feature Importance")
        sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
        x_feats = [x[0] for x in sorted_feats]
        y_imp = [x[1] for x in sorted_feats]
        
        fig_imp = px.bar(
            x=y_imp, y=x_feats, orientation='h',
            labels={'x': 'Importance Score', 'y': 'Feature'},
            title=f"Feature Impact ({meta['best_model']})",
            template="plotly_dark",
            color=y_imp,
            color_continuous_scale="Mint"
        )
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_imp, use_container_width=True)
        st.info("Rainfall and humidity carry the most weight in the model's decision, making water availability the key factor for these crops.")


# --- PAGE 3: CROP RECOMMENDATION TOOL ---
elif page == "Crop Prediction":
    st.title("Crop Prediction Model")
    st.markdown("Enter the soil and environmental data below to see which crop the model recommends.")
    
    with st.form("prediction_form"):
        st.subheader("Soil Properties")
        col1, col2, col3 = st.columns(3)
        with col1:
            N = st.number_input("Nitrogen (N) [ratio]", min_value=0, max_value=200, value=50)
        with col2:
            P = st.number_input("Phosphorus (P) [ratio]", min_value=0, max_value=200, value=50)
        with col3:
            K = st.number_input("Potassium (K) [ratio]", min_value=0, max_value=250, value=50)
            
        st.subheader("Environmental Factors")
        col4, col5, col6, col7 = st.columns(4)
        with col4:
            temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
        with col5:
            hum = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
        with col6:
            ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        with col7:
            rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, step=1.0)
            
        submit_button = st.form_submit_button("Predict Crop", type="primary")

    if submit_button:
        with st.spinner("Running prediction..."):
            # Prepare input
            input_data = pd.DataFrame([[N, P, K, temp, hum, ph, rain]], columns=meta['features'])
            input_scaled = scaler.transform(input_data)
            
            # Predict
            pred_encoded = model.predict(input_scaled)[0]
            pred_crop = le.inverse_transform([pred_encoded])[0].capitalize()
            
            # Probabilities
            probs = model.predict_proba(input_scaled)[0]
            max_prob = np.max(probs) * 100
            
            # Display Result
            st.markdown(f"""
            <div class="success-box">
                <p style="margin:0; font-size: 18px;">Recommended crop for these conditions:</p>
                <p class="pred-text">{pred_crop}</p>
                <p class="conf-text">Model Confidence: {max_prob:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show top 3 probabilities
            st.markdown("#### Next Best Alternatives")
            top_3_indices = np.argsort(probs)[-3:][::-1]
            
            top_3_crops = [le.inverse_transform([i])[0].capitalize() for i in top_3_indices]
            top_3_probs = [probs[i] * 100 for i in top_3_indices]
            
            for crop, prob in zip(top_3_crops, top_3_probs):
                if crop != pred_crop and prob > 0:
                    st.progress(prob / 100, text=f"{crop} ({prob:.2f}%)")
