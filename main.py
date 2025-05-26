import streamlit as st
import pandas as pd
from pipeline import inference_pipeline

# ----------------------------
# 1) Load data at startup
# ----------------------------
# @st.cache_data
def load_data():
    patient_data = pd.read_csv("data/patient_data.csv")
    return patient_data

all_data = load_data()

# ----------------------------
# 2) Helper function to categorize glucose
# ----------------------------
def categorize_glucose(glucose_level: float) -> str:
    if pd.isna(glucose_level):
        return 'Unknown'
    elif glucose_level < 3.9:
        return 'Low'
    elif glucose_level <= 7.8:
        return 'Normal'
    elif glucose_level <= 10.0:
        return 'Slightly High'
    elif glucose_level <= 13.9:
        return 'High'
    else:
        return 'Critical High'

# ----------------------------
# 3) Run inference pipeline
# ----------------------------
def get_patients(df):
    df200 = df.iloc[:200].copy()
    if "bg+1:00" in df200.columns:
        df200 = df200.drop(["bg+1:00"], axis=1)

    pipeline_input = df200.drop(columns=["p_num"], errors="ignore")
    predicted_bgs = inference_pipeline(pipeline_input)

    patients = []
    for i, bg in enumerate(predicted_bgs):
        category = categorize_glucose(bg)
        if category != "Normal":
            patients.append({
                "id": df200.at[i, "id"],
                "p_num": df200.at[i, "p_num"],
                "glucose": float(bg),
                "critical_level": category,
            })
    return patients

# ----------------------------
# 4) Enhanced Streamlit UI
# ----------------------------
st.set_page_config(
    page_title="Clinical Glucose Monitor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for medical dashboard styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }

    .main-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }

    .subtitle {
        color: #e0e7ff;
        font-size: 1.2rem;
        text-align: center;
        margin-top: 0.5rem;
    }

    .stats-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 15px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        border-left: 5px solid #3b82f6;
    }

    .patient-card {
        background: white;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-left: 4px solid;
        transition: transform 0.2s ease;
    }

    .patient-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    }

    .critical-card { border-left-color: #dc2626; }
    .high-card { border-left-color: #f59e0b; }
    .low-card { border-left-color: #2563eb; }

    .patient-id {
        font-weight: 700;
        font-size: 1.1rem;
        color: #1f2937;
    }

    .glucose-value {
        font-size: 1.3rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    .critical-value { color: #dc2626; }
    .high-value { color: #f59e0b; }
    .low-value { color: #2563eb; }

    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        color: white;
    }

    .critical-header { background: linear-gradient(45deg, #dc2626, #ef4444); }
    .high-header { background: linear-gradient(45deg, #f59e0b, #fbbf24); }
    .low-header { background: linear-gradient(45deg, #2563eb, #3b82f6); }

    .metric-box {
        text-align: center;
        padding: 1rem;
        background: #f8fafc;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    .metric-number {
        font-size: 2rem;
        font-weight: 800;
        color: #1f2937;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .alert-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .critical-badge { background: #fee2e2; color: #dc2626; }
    .high-badge { background: #fef3c7; color: #f59e0b; }
    .low-badge { background: #dbeafe; color: #2563eb; }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🏥 Clinical Glucose Monitoring System</h1>
    <p class="subtitle">Real-time Patient Blood Glucose Level Dashboard</p>
</div>
""", unsafe_allow_html=True)

# Get patient data
patients = get_patients(all_data)

# Statistics Overview
critical_count = len([p for p in patients if p['critical_level'] == 'Critical High'])
high_count = len([p for p in patients if p['critical_level'] in ['High', 'Slightly High']])
low_count = len([p for p in patients if p['critical_level'] == 'Low'])
total_alerts = len(patients)

st.markdown("""
<div class="stats-container">
    <h3 style="color: #1f2937; margin-bottom: 1rem;">📊 Alert Summary</h3>
</div>
""", unsafe_allow_html=True)

# Metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-number" style="color: #dc2626;">{critical_count}</div>
        <div class="metric-label">Critical High</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-number" style="color: #f59e0b;">{high_count}</div>
        <div class="metric-label">High Levels</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-number" style="color: #2563eb;">{low_count}</div>
        <div class="metric-label">Low Levels</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-number" style="color: #1f2937;">{total_alerts}</div>
        <div class="metric-label">Total Alerts</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Patient Alert Sections
critical_col, high_col, low_col = st.columns(3)

with critical_col:
    st.markdown("""
    <div class="section-header critical-header">
        🚨 CRITICAL HIGH
    </div>
    """, unsafe_allow_html=True)

    critical_patients = [p for p in patients if p['critical_level'] == 'Critical High']
    if critical_patients:
        for p in critical_patients:
            st.markdown(f"""
            <div class="patient-card critical-card">
                <div class="patient-id">Patient ID: {p['id']}</div>
                <div style="color: #6b7280; font-size: 0.9rem;">P-Number: {p['p_num']}</div>
                <div class="glucose-value critical-value">{p['glucose']:.2f} mg/dL</div>
                <span class="alert-badge critical-badge">Critical Alert</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #6b7280;">
            <p>✅ No critical high glucose levels detected</p>
        </div>
        """, unsafe_allow_html=True)

with high_col:
    st.markdown("""
    <div class="section-header high-header">
        ⚠️ HIGH LEVELS
    </div>
    """, unsafe_allow_html=True)

    high_patients = [p for p in patients if p['critical_level'] in ['High', 'Slightly High']]
    if high_patients:
        for p in high_patients:
            badge_text = "High Alert" if p['critical_level'] == 'High' else "Elevated"
            st.markdown(f"""
            <div class="patient-card high-card">
                <div class="patient-id">Patient ID: {p['id']}</div>
                <div style="color: #6b7280; font-size: 0.9rem;">P-Number: {p['p_num']}</div>
                <div class="glucose-value high-value">{p['glucose']:.2f} mg/dL</div>
                <span class="alert-badge high-badge">{badge_text}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #6b7280;">
            <p>✅ No high glucose levels detected</p>
        </div>
        """, unsafe_allow_html=True)

with low_col:
    st.markdown("""
    <div class="section-header low-header">
        🔵 LOW LEVELS
    </div>
    """, unsafe_allow_html=True)

    low_patients = [p for p in patients if p['critical_level'] == 'Low']
    if low_patients:
        for p in low_patients:
            st.markdown(f"""
            <div class="patient-card low-card">
                <div class="patient-id">Patient ID: {p['id']}</div>
                <div style="color: #6b7280; font-size: 0.9rem;">P-Number: {p['p_num']}</div>
                <div class="glucose-value low-value">{p['glucose']:.2f} mg/dL</div>
                <span class="alert-badge low-badge">Low Alert</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #6b7280;">
            <p>✅ No low glucose levels detected</p>
        </div>
        """, unsafe_allow_html=True)

# Footer with reference ranges
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="background: #f8fafc; padding: 1.5rem; border-radius: 10px; border: 1px solid #e5e7eb;">
    <h4 style="color: #374151; margin-bottom: 1rem;">📋 Glucose Level Reference Ranges</h4>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
        <div><strong>Normal:</strong> 3.9 - 7.8 mg/dL</div>
        <div><strong>Slightly High:</strong> 7.9 - 10.0 mg/dL</div>
        <div><strong>High:</strong> 10.1 - 13.9 mg/dL</div>
        <div><strong>Critical High:</strong> > 13.9 mg/dL</div>
        <div><strong>Low:</strong> < 3.9 mg/dL</div>
    </div>
</div>
""", unsafe_allow_html=True)