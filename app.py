import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from fpdf import FPDF

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title=" Soil Health & Farmer Awareness Dashboard ",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD DATA AND TRAIN MODEL ---
@st.cache_data
def load_data_and_model():
    # Build robust absolute path to dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'Crop_recommendation.csv')
    
    df = pd.read_csv(csv_path)
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return df, model

df, model = load_data_and_model()

# --- HELPER FUNCTIONS ---
def evaluate_soil_health(N, P, K, ph):
    """Evaluates soil quality and provides actionable recommendations."""
    score = 100
    status = []
    recs = []
    
    # Nitrogen Analysis
    if N < 50:
        score -= 15
        status.append("Low Nitrogen (N)")
        recs.append("Apply nitrogen-rich fertilizers like Urea or add composted farmyard manure.")
    elif N > 120:
        score -= 5
        status.append("High Nitrogen (N)")
        recs.append("Reduce nitrogen application to avoid leaf burn and runoff.")

    # Phosphorus Analysis
    if P < 30:
        score -= 15
        status.append("Low Phosphorus (P)")
        recs.append("Apply Single Super Phosphate (SSP) or DAP (Di-ammonium Phosphate) before sowing.")
    elif P > 80:
        score -= 5
        status.append("High Phosphorus (P)")

    # Potassium Analysis
    if K < 30:
        score -= 15
        status.append("Low Potassium (K)")
        recs.append("Apply Muriate of Potash (MOP) or Potash-rich organic fertilizers.")
    elif K > 80:
        score -= 5
        status.append("High Potassium (K)")

    # pH Analysis
    if ph < 5.5:
        score -= 20
        status.append("Acidic Soil")
        recs.append("Apply agricultural lime (calcium carbonate) to increase pH to neutral levels.")
    elif ph > 7.5:
        score -= 20
        status.append("Alkaline Soil")
        recs.append("Apply gypsum or organic matter (compost) to lower pH levels.")
    else:
        status.append("Optimal pH")

    if score >= 80:
        quality = "Excellent Soil Health"
    elif score >= 60:
        quality = "Good Soil Health"
    else:
        quality = "Needs Soil Conditioning"

    return score, quality, status, recs


def get_top_crops(model, user_features, top_n=3):
    """Returns top N crop predictions with their confidence probabilities."""
    probs = model.predict_proba(user_features)[0]
    classes = model.classes_
    
    # Sort indices by highest probability
    top_indices = np.argsort(probs)[::-1][:top_n]
    
    top_crops = []
    for idx in top_indices:
        crop_name = classes[idx]
        confidence = probs[idx] * 100
        top_crops.append((crop_name, confidence))
        
    return top_crops


def generate_pdf_report(n, p, k, temp, hum, ph, rain, top_crops, score, quality, recs):
    """Generates a downloadable PDF report."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    
    # Title
    pdf.cell(200, 10, txt="Soil Health & Crop Suitability Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="---------------------------------------------------------", ln=True, align='C')
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="1. Input Soil Parameters:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 6, txt=f" - Nitrogen (N): {n} kg/ha | Phosphorus (P): {p} kg/ha | Potassium (K): {k} kg/ha", ln=True)
    pdf.cell(200, 6, txt=f" - pH Level: {ph} | Temperature: {temp} C | Humidity: {hum}% | Rainfall: {rain} mm", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="2. Soil Health Assessment:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 6, txt=f" - Health Score: {score}/100 ({quality})", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="3. Top Recommended Crops:", ln=True)
    pdf.set_font("Arial", size=10)
    for rank, (crop, prob) in enumerate(top_crops, start=1):
        pdf.cell(200, 6, txt=f" - #{rank} {crop.upper()} ({prob:.1f}% Match Suitability)", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="4. Actionable Soil Improvement Recommendations:", ln=True)
    pdf.set_font("Arial", size=10)
    if recs:
        for r in recs:
            pdf.multi_cell(0, 6, txt=f" - {r}")
    else:
        pdf.cell(200, 6, txt=" - Your soil nutrients are optimal. Maintain current organic management practices.", ln=True)
        
    return pdf.output(dest='S').encode('latin-1')


# --- HEADER ---
st.title("🌱 Soil Health Data Analysis & Farmer Advisory System")
st.markdown("""
This platform helps farmers analyze soil health status, evaluate suitability for multiple crops, 
receive step-by-step improvement recommendations, and download a tailored soil test report.
""")

st.divider()

# --- SIDEBAR INPUT FORM ---
st.sidebar.header("🧪 Enter Soil Test Parameters")

input_n = st.sidebar.number_input("Nitrogen (N) [kg/ha]", min_value=0, max_value=200, value=75)
input_p = st.sidebar.number_input("Phosphorus (P) [kg/ha]", min_value=0, max_value=200, value=45)
input_k = st.sidebar.number_input("Potassium (K) [kg/ha]", min_value=0, max_value=200, value=50)
input_ph = st.sidebar.slider("Soil pH Level", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

st.sidebar.header("🌤️ Environmental Conditions")
input_temp = st.sidebar.slider("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
input_hum = st.sidebar.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
input_rain = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=150.0)

# Feature matrix for prediction
user_features = np.array([[input_n, input_p, input_k, input_temp, input_hum, input_ph, input_rain]])

# Get Top 3 Recommended Crops
top_crops = get_top_crops(model, user_features, top_n=3)
primary_crop = top_crops[0][0]

# Evaluate Soil Health Score & Recommendations
score, quality, status_list, recommendations = evaluate_soil_health(input_n, input_p, input_k, input_ph)

# --- DASHBOARD LAYOUT (TABS) ---
tab1, tab2, tab3 = st.tabs(["📊 Dynamic Analysis & Recommendations", "📈 Dataset Insights & Charts", "📄 Download Health Report"])

# --- TAB 1: USER ANALYSIS & RECOMMENDATIONS ---
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🌾 Suitable Crops for Your Soil")
        
        # Metric display for soil health
        st.metric(label="Overall Soil Health Score", value=f"{score} / 100", delta=quality)
        
        # Top 1 Crop
        st.success(f"**#1 Primary Recommended Crop:**  \n### 🌽 **{primary_crop.upper()}** ({top_crops[0][1]:.1f}% Match)")
        
        # Alternative Crops
        st.markdown("##### 💡 Alternative Crop Options:")
        for rank, (crop, prob) in enumerate(top_crops[1:], start=2):
            st.write(f"**#{rank} {crop.capitalize()}** — `{prob:.1f}% Match`")
            st.progress(min(int(prob), 100))
        
        st.markdown("---")
        st.markdown("**Soil Status Summary:**")
        for status in status_list:
            st.write(f"- {status}")
            
    with col2:
        st.subheader("📊 Your Soil N-P-K Levels vs Optimal Averages")
        
        # Compare user's soil vs ideal NPK for primary predicted crop
        crop_avg = df[df['label'] == primary_crop][['N', 'P', 'K']].mean()
        
        categories = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
        user_vals = [input_n, input_p, input_k]
        ideal_vals = [crop_avg['N'], crop_avg['P'], crop_avg['K']]
        
        fig_npk = go.Figure()
        fig_npk.add_trace(go.Bar(x=categories, y=user_vals, name='Your Soil Levels', marker_color='#2b5c28'))
        fig_npk.add_trace(go.Bar(x=categories, y=ideal_vals, name=f'Ideal for {primary_crop.capitalize()}', marker_color='#aed581'))
        
        fig_npk.update_layout(barmode='group', title_text="Nutrient Comparison (kg/ha)", height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_npk, use_container_width=True)

    st.divider()
    
    st.subheader("🛠️ Step-by-Step Soil Health Improvement Recommendations")
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.info(f"**Recommendation {i}:** {rec}")
    else:
        st.success("✨ Your soil nutrients and pH level are well balanced! Continue practicing sustainable crop rotation and adding organic compost.")

# --- TAB 2: DATASET INSIGHTS & INTERACTIVE VISUALIZATIONS ---
with tab2:
    st.subheader("📌 Visual Insights for Farmers & Agricultural Awareness")
    
    vcol1, vcol2 = st.columns(2)
    
    with vcol1:
        st.markdown("##### 1. Average Soil pH Requirements by Crop")
        fig_ph = px.box(df, x='label', y='ph', color='label', title="Soil pH Spread Across Crops")
        fig_ph.update_layout(showlegend=False, xaxis_title="Crop", yaxis_title="pH")
        st.plotly_chart(fig_ph, use_container_width=True)
        
    with vcol2:
        st.markdown("##### 2. Rainfall vs Temperature Requirements")
        fig_scat = px.scatter(
            df, x='temperature', y='rainfall', color='label', 
            hover_data=['N', 'P', 'K', 'ph'],
            title="Temperature & Rainfall Cluster Analysis for Crops"
        )
        st.plotly_chart(fig_scat, use_container_width=True)

    st.markdown("##### 3. Average N-P-K Demands Across All Dataset Crops")
    df_avg = df.groupby('label')[['N', 'P', 'K']].mean().reset_index()
    fig_avg = px.bar(df_avg, x='label', y=['N', 'P', 'K'], title="Nutrient Demands by Crop", barmode='group')
    fig_avg.update_layout(xaxis_title="Crop", yaxis_title="Nutrient Level (kg/ha)")
    st.plotly_chart(fig_avg, use_container_width=True)

# --- TAB 3: DOWNLOAD REPORT ---
with tab3:
    st.subheader("📥 Downloadable Soil Health Analysis Report")
    st.write("Generate an official, simple summary PDF report containing your input test values, soil quality rating, top crop predictions, and improvement steps.")
    
    pdf_bytes = generate_pdf_report(
        input_n, input_p, input_k, input_temp, input_hum, input_ph, input_rain,
        top_crops, score, quality, recommendations
    )
    
    st.download_button(
        label="📄 Download Soil Health Report (PDF)",
        data=pdf_bytes,
        file_name="Soil_Health_Report.pdf",
        mime="application/pdf"
    )
