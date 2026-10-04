
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import time

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & MODERN DARK UI STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Defect Detection AI", 
    layout="wide", 
    page_icon="🏭",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Dark Mode & Animated Components
st.markdown("""
    <style>
    /* Global Page Background */
    .stApp {
        background: radial-gradient(circle at top left, #0f172a 0%, #020617 100%);
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Modern Sidebar Dark Theme */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Header Animations & Gradient Text */
    @keyframes fadeInDown {
        0% { opacity: 0; transform: translateY(-20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .main-title {
        animation: fadeInDown 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        text-align: center;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.2rem;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* Glassmorphism Dark Cards */
    .css-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 20px;
    }
    .css-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
        box-shadow: 0 20px 40px -15px rgba(56, 189, 248, 0.15);
        transform: translateY(-2px);
    }

    /* Styled Prediction Alert Boxes */
    @keyframes slideInUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .prediction-box {
        animation: slideInUp 0.6s ease-out;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-top: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.4) 100%);
        color: #34d399;
        border-color: #10b981;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.25);
    }

    .warning-box {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.4) 100%);
        color: #fbbf24;
        border-color: #f59e0b;
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.25);
    }

    .danger-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.4) 100%);
        color: #f87171;
        border-color: #ef4444;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.25);
    }

    /* Dark Mode Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: #ffffff;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #60a5fa 0%, #2563eb 100%);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
        transform: translateY(-1px);
    }

    /* Clean Dividers & Headings */
    h1, h2, h3, h4 {
        color: #f1f5f9 !important;
    }
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. LOAD MODEL AND DATA
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load('defect_model.pkl')
        encoders = joblib.load('encoders.pkl')
        return model, encoders
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run `python ml_model.py` first to train the model.")
        st.stop()

@st.cache_data
def load_data():
    df = pd.read_csv('defects_data.csv')
    return df

model, encoders = load_model()
df = load_data()

# ---------------------------------------------------------
# 3. UI LAYOUT & CONTROLS
# ---------------------------------------------------------
st.markdown('<div class="main-title">🏭 AI Defect Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predictive Quality Analysis</div>', unsafe_allow_html=True)

# Sidebar for User Inputs
st.sidebar.markdown("### 🎛️ Control Panel")
st.sidebar.caption("Adjust manufacturing metrics below to predict defects.")

def user_input_features():
    defect_location = st.sidebar.selectbox('Defect Location', df['defect_location'].unique())
    severity = st.sidebar.selectbox('Severity', df['severity'].unique())
    inspection_method = st.sidebar.selectbox('Inspection Method', df['inspection_method'].unique())
    repair_cost = st.sidebar.slider('Repair Cost ($)', float(df['repair_cost'].min()), float(df['repair_cost'].max()), 500.0)
    
    # Date features
    month = st.sidebar.slider('Month', 1, 12, 6)
    day_of_week = st.sidebar.slider('Day of Week (0=Mon, 6=Sun)', 0, 6, 2)
    
    data = {
        'defect_location': defect_location,
        'severity': severity,
        'inspection_method': inspection_method,
        'repair_cost': repair_cost,
        'Month': month,
        'DayOfWeek': day_of_week
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Main Layout
col1, col2 = st.columns([1, 1.4], gap="medium")

with col1:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("📋 Input Overview")
    st.dataframe(input_df, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Prediction Trigger Button
    if st.button("🔍 Run Defect Prediction", use_container_width=True):
        with st.spinner("Processing telemetry data..."):
            time.sleep(1.2) # Smooth animation effect
            
            # Preprocess Input
            input_processed = input_df.copy()
            for col in ['defect_location', 'severity', 'inspection_method']:
                input_processed[col] = encoders[col].transform(input_processed[col])
            
            # Run Inference
            prediction_encoded = model.predict(input_processed)
            prediction_proba = model.predict_proba(input_processed)
            prediction_label = encoders['target'].inverse_transform(prediction_encoded)[0]
            confidence = np.max(prediction_proba) * 100
            
            # Render Styled Results
            if prediction_label == "Structural":
                st.balloons()
                st.markdown(f'<div class="prediction-box success-box">✅ Predicted Defect: {prediction_label}</div>', unsafe_allow_html=True)
            elif prediction_label == "Functional":
                st.markdown(f'<div class="prediction-box warning-box">⚠️ Predicted Defect: {prediction_label}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-box danger-box">🚨 Predicted Defect: {prediction_label}</div>', unsafe_allow_html=True)
            
            st.markdown(f"<p style='text-align: center; margin-top: 10px; color: #94a3b8;'>Model Confidence: <strong style='color: #38bdf8;'>{confidence:.2f}%</strong></p>", unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("📊 Exploratory Insights")
    
    tab1, tab2, tab3 = st.tabs(["Defect Types", "Severity vs Cost", "Inspection Methods"])
    
    # Helper layout styling for Plotly dark mode
    def apply_dark_plotly_style(fig):
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", family="Inter, sans-serif"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    with tab1:
        fig1 = px.pie(
            df, names='defect_type', title='Distribution of Defect Types', 
            hole=0.5, color_discrete_sequence=px.colors.qualitative.Dark24
        )
        st.plotly_chart(apply_dark_plotly_style(fig1), use_container_width=True)
        
    with tab2:
        fig2 = px.box(
            df, x='severity', y='repair_cost', color='severity', 
            title='Repair Cost by Severity Level', color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(apply_dark_plotly_style(fig2), use_container_width=True)
        
    with tab3:
        fig3 = px.histogram(
            df, x='inspection_method', color='defect_type', barmode='group', 
            title='Inspection Method vs Defect Types', color_discrete_sequence=px.colors.qualitative.Prism
        )
        st.plotly_chart(apply_dark_plotly_style(fig3), use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.9rem;'>Built by team 8 | Powered by Streamlit & Scikit-Learn</p>", unsafe_allow_html=True)

















































































































































































































# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import plotly.express as px
# import time

# # ---------------------------------------------------------
# # 1. PAGE CONFIGURATION & MODERN CSS ANIMATIONS
# # ---------------------------------------------------------
# st.set_page_config(page_title="Defect Detection AI", layout="wide", page_icon="🏭")

# # Custom CSS for Modern UI and Animations
# st.markdown("""
#     <style>
#     /* Main Background Gradient */
#     .stApp {
#         background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
#         font-family: 'Inter', sans-serif;
#     }
    
#     /* Title Animation */
#     @keyframes fadeInDown {
#         0% { opacity: 0; transform: translateY(-20px); }
#         100% { opacity: 1; transform: translateY(0); }
#     }
#     .main-title {
#         animation: fadeInDown 1s ease-out;
#         text-align: center;
#         color: #1e3a8a;
#         font-weight: 800;
#         font-size: 3rem;
#         margin-bottom: 0.5rem;
#     }
    
#     .sub-title {
#         text-align: center;
#         color: #4b5563;
#         font-size: 1.2rem;
#         margin-bottom: 2rem;
#     }

#     /* Card Styling */
#     .css-card {
#         background: rgba(255, 255, 255, 0.9);
#         padding: 20px;
#         border-radius: 15px;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#         transition: transform 0.3s ease, box-shadow 0.3s ease;
#         margin-bottom: 20px;
#     }
#     .css-card:hover {
#         transform: translateY(-5px);
#         box-shadow: 0 10px 15px rgba(0, 0, 0, 0.2);
#     }

#     /* Prediction Result Animation */
#     @keyframes slideInUp {
#         0% { opacity: 0; transform: translateY(30px); }
#         100% { opacity: 1; transform: translateY(0); }
#     }
#     .prediction-box {
#         animation: slideInUp 0.8s ease-out;
#         padding: 20px;
#         border-radius: 10px;
#         text-align: center;
#         font-size: 1.5rem;
#         font-weight: bold;
#         color: white;
#         margin-top: 20px;
#     }
    
#     /* Success Pulse Animation */
#     @keyframes pulse {
#         0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
#         70% { box-shadow: 0 0 0 15px rgba(16, 185, 129, 0); }
#         100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
#     }
#     .pulse-success {
#         animation: pulse 2s infinite;
#     }
    
#     /* Error Shake Animation */
#     @keyframes shake {
#         0% { transform: translateX(0); }
#         25% { transform: translateX(-5px); }
#         50% { transform: translateX(5px); }
#         75% { transform: translateX(-5px); }
#         100% { transform: translateX(0); }
#     }
#     .shake-error {
#         animation: shake 0.5s ease-in-out;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # ---------------------------------------------------------
# # 2. LOAD MODEL AND DATA
# # ---------------------------------------------------------
# @st.cache_resource
# def load_model():
#     try:
#         model = joblib.load('defect_model.pkl')
#         encoders = joblib.load('encoders.pkl')
#         return model, encoders
#     except FileNotFoundError:
#         st.error("⚠️ Model files not found! Please run `python ml_model.py` first to train the model.")
#         st.stop()

# @st.cache_data
# def load_data():
#     df = pd.read_csv('defects_data.csv')
#     return df

# model, encoders = load_model()
# df = load_data()

# # ---------------------------------------------------------
# # 3. UI LAYOUT
# # ---------------------------------------------------------
# st.markdown('<div class="main-title">🏭 AI Defect Detection System</div>', unsafe_allow_html=True)
# st.markdown('<div class="sub-title">DMDW Lab Project | Predictive Quality Analysis</div>', unsafe_allow_html=True)

# # Sidebar for User Inputs
# st.sidebar.markdown("### 🎛️ Input Parameters")
# st.sidebar.info("Adjust the parameters to predict the type of manufacturing defect.")

# def user_input_features():
#     defect_location = st.sidebar.selectbox('Defect Location', df['defect_location'].unique())
#     severity = st.sidebar.selectbox('Severity', df['severity'].unique())
#     inspection_method = st.sidebar.selectbox('Inspection Method', df['inspection_method'].unique())
#     repair_cost = st.sidebar.slider('Repair Cost ($)', float(df['repair_cost'].min()), float(df['repair_cost'].max()), 500.0)
    
#     # Date features
#     month = st.sidebar.slider('Month (1-12)', 1, 12, 6)
#     day_of_week = st.sidebar.slider('Day of Week (0=Mon, 6=Sun)', 0, 6, 2)
    
#     data = {
#         'defect_location': defect_location,
#         'severity': severity,
#         'inspection_method': inspection_method,
#         'repair_cost': repair_cost,
#         'Month': month,
#         'DayOfWeek': day_of_week
#     }
#     features = pd.DataFrame(data, index=[0])
#     return features

# input_df = user_input_features()

# # Main Content Area
# col1, col2 = st.columns([1, 1.5])

# with col1:
#     st.markdown('<div class="css-card">', unsafe_allow_html=True)
#     st.subheader("📋 Input Summary")
#     st.dataframe(input_df, use_container_width=True)
#     st.markdown('</div>', unsafe_allow_html=True)

#     # Prediction Button
#     if st.button("🔍 Predict Defect Type", use_container_width=True):
#         with st.spinner("Analyzing data..."):
#             time.sleep(1.5) # Simulate processing time for animation effect
            
#             # Preprocess input
#             input_processed = input_df.copy()
#             for col in ['defect_location', 'severity', 'inspection_method']:
#                 input_processed[col] = encoders[col].transform(input_processed[col])
            
#             # Make prediction
#             prediction_encoded = model.predict(input_processed)
#             prediction_proba = model.predict_proba(input_processed)
#             prediction_label = encoders['target'].inverse_transform(prediction_encoded)[0]
            
#             # Display Result
#             if prediction_label == "Structural":
#                 st.balloons()
#                 st.markdown(f'<div class="prediction-box pulse-success" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">✅ Predicted Defect: {prediction_label}</div>', unsafe_allow_html=True)
#             elif prediction_label == "Functional":
#                 st.markdown(f'<div class="prediction-box" style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);">⚠️ Predicted Defect: {prediction_label}</div>', unsafe_allow_html=True)
#             else:
#                 st.markdown(f'<div class="prediction-box shake-error" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);">🚨 Predicted Defect: {prediction_label}</div>', unsafe_allow_html=True)
            
#             st.write(f"**Confidence:** {np.max(prediction_proba) * 100:.2f}%")

# with col2:
#     st.markdown('<div class="css-card">', unsafe_allow_html=True)
#     st.subheader("📊 Exploratory Data Analysis")
    
#     # Interactive Plotly Charts
#     tab1, tab2, tab3 = st.tabs(["Defect Types", "Severity vs Cost", "Inspection Methods"])
    
#     with tab1:
#         fig1 = px.pie(df, names='defect_type', title='Distribution of Defect Types', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
#         st.plotly_chart(fig1, use_container_width=True)
        
#     with tab2:
#         fig2 = px.box(df, x='severity', y='repair_cost', color='severity', title='Repair Cost by Severity', color_discrete_sequence=px.colors.qualitative.Set2)
#         st.plotly_chart(fig2, use_container_width=True)
        
#     with tab3:
#         fig3 = px.histogram(df, x='inspection_method', color='defect_type', barmode='group', title='Inspection Methods vs Defect Types', color_discrete_sequence=px.colors.qualitative.Bold)
#         st.plotly_chart(fig3, use_container_width=True)
        
#     st.markdown('</div>', unsafe_allow_html=True)

# # Footer
# st.markdown("---")
# st.markdown("<p style='text-align: center; color: #6b7280;'>Built for DMDW Lab | Powered by Streamlit & Scikit-Learn</p>", unsafe_allow_html=True)