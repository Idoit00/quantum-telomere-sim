"""
Quantum Telomere Simulation
Uses a multi-qubit quantum circuit to model biological age uncertainty.
Includes: Age, Stress, Strength Training, Body Fat % Change, Diet Quality.
Research math: 0.67 bp/min (strength), 4% decay for weight gain >15%.
"""

import math
import streamlit as st
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_bloch_vector
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

DATA_PATH = "study_data.csv"


def run_quantum_simulation(age, stress, exercise_mins, weight_gain, diet_quality, muscle_score):
    """Research-backed quantum simulation: multi-qubit register driven by CSV values."""
    df = pd.read_csv(DATA_PATH)

    # Extract study values (with fallbacks if factor missing)
    def get_effect(factor, default):
        row = df[df["Factor"] == factor]
        return float(row["Effect_Size"].values[0]) if len(row) > 0 else default

    strength_impact = get_effect("Strength_Training", 0.67)
    diet_impact = get_effect("Sugar_Meat_Diet", -50.0)
    weight_impact = get_effect("Adiposity", -4.0)
    inheritance_base = get_effect("Heritability", 64.0) / 100.0
    noise_level = (age / 100.0) * 0.1

    # Q0 (Genetics): theta based on age
    theta_q0 = (age / 100.0) * math.pi
    metabolic_milestone = (39 <= age <= 42) or (69 <= age <= 72)
    if metabolic_milestone:
        # Significant TL velocity changes are observed near ages 40 and 70.
        theta_q0 += 0.15 * math.pi
    theta_q0 = max(0.0, min(math.pi, theta_q0))

    # Q1/Q2/Q3: rotations from study effect sizes
    theta_q1 = max(0.0, min(math.pi, (0.5 + (strength_impact / 100.0)) * math.pi))
    theta_q2 = max(0.0, min(math.pi, (0.5 + (diet_impact / 100.0)) * math.pi))
    theta_q3 = max(0.0, min(math.pi, (0.5 + (weight_impact / 100.0)) * math.pi))

    # Run the circuit: 4-qubit register, 1 classical bit
    qc = QuantumCircuit(4, 1)
    qc.ry(theta_q0, 0)
    qc.ry(theta_q1, 1)
    qc.ry(theta_q2, 2)
    qc.ry(theta_q3, 3)

    # Entanglement links
    qc.cx(1, 0)  # Exercise -> Genetics
    qc.cx(2, 3)  # Diet -> Adiposity

    # Homeostatic failure tipping points under extreme environmental pressure.
    if stress > 80:
        # High stress induces a bit-flip that overrides baseline genetic stability.
        qc.x(0)
    if exercise_mins < 30:
        # Sedentary state increases instability coupling between genetics and movement.
        qc.cx(0, 1)

    elite_longevity = diet_quality > 85 and exercise_mins > 180
    metabolic_shield = muscle_score > 70 and weight_gain < 5
    if elite_longevity:
        # Blue Zone synergy uses Toffoli-style correction: when exercise + diet controls align,
        # the circuit applies quantum error correction to counter noise in genetics.
        qc.ccx(1, 2, 0)
        qc.ry(-0.12, 0)  # Nudge Q0 back toward |0> (younger/resilient state)

    # Decoherence layer: age-driven biological entropy across all qubits
    for i in range(4):
        qc.rx(noise_level, i)
        qc.rz(noise_level, i)

    # Measure Genetics qubit only
    qc.measure(0, 0)

    sim = Aer.get_backend("qasm_simulator")
    compiled = transpile(qc, sim)
    job = sim.run(compiled, shots=1024)
    result = job.result()
    counts = result.get_counts()

    n = 1024
    prob_0 = counts.get("0", 0) / n
    prob_1 = counts.get("1", 0) / n

    # Biological age from quantum result + inputs
    decay = prob_1
    entropy_impact = prob_1 * (age / 20.0)
    threshold_penalty = 0.0
    if stress > 80:
        threshold_penalty += 5.0
    if exercise_mins < 30:
        threshold_penalty += 3.0
    tipping_point_reached = threshold_penalty > 0.0
    blue_zone_bonus = 0.0
    if elite_longevity:
        blue_zone_bonus += 4.0
    if metabolic_shield:
        blue_zone_bonus += 2.5
    blue_zone_active = blue_zone_bonus > 0.0
    velocity_penalty = 2.5 if metabolic_milestone else 0.0
    bio_age = age + (decay * 8) - (exercise_mins * strength_impact / 500.0)
    if weight_gain > 15:
        bio_age += abs(weight_impact) / 5.0
    bio_age += (1.0 - muscle_score / 100.0) * 3.0
    bio_age -= (inheritance_base - 0.5) * 1.5
    bio_age += entropy_impact
    bio_age += threshold_penalty
    bio_age -= blue_zone_bonus
    bio_age += velocity_penalty
    bio_age = round(max(20, min(95, bio_age)), 1)

    return {
        "theta": theta_q0,
        "theta_components": {
            "q0_genetics_age": theta_q0,
            "q1_exercise": theta_q1,
            "q2_nutrition": theta_q2,
            "q3_adiposity": theta_q3,
        },
        "prob_0": prob_0,
        "prob_1": prob_1,
        "entropy_level": noise_level,
        "tipping_point_reached": tipping_point_reached,
        "blue_zone_active": blue_zone_active,
        "metabolic_milestone": metabolic_milestone,
        "bloch_coords": [math.sin(theta_q0), 0, math.cos(theta_q0)],
        "biological_age": bio_age,
    }


def run_quantum_research_sim(age, stress, exercise_mins, weight_gain, diet_quality, muscle_score):
    """Backward-compatible wrapper."""
    return run_quantum_simulation(age, stress, exercise_mins, weight_gain, diet_quality, muscle_score)


def display_bloch_sphere(coords):
    """Render a Bloch sphere for the Genetics qubit statevector projection."""
    fig = plot_bloch_vector(coords)
    fig.patch.set_facecolor("#1a1a1a")
    fig.patch.set_alpha(0.0)
    for ax in fig.axes:
        ax.set_facecolor("#1a1a1a")
    return fig


# Page config - must be first Streamlit command
st.set_page_config(
    page_title="Quantum Telomere Simulation",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS - Dark mode, white text, glassmorphism
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&display=swap');
    
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
    section[data-testid="stSidebar"] > div, .block-container {
        background-color: #1a1a1a !important;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown, .stCaption {
        color: #FFFFFF !important;
    }
    
    .main-header {
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 2rem;
        font-weight: 600;
        color: #FFFFFF !important;
        border-bottom: 2px solid #404040;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    .subtitle {
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 1rem;
        color: #FFFFFF !important;
        margin-bottom: 2rem;
    }
    
    .glass-card, .metric-card {
        background: rgba(15, 15, 20, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
    }
    
    .metric-card { text-align: center; }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #FFFFFF !important;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #b0b0b0 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
    
    .bio-age-readout {
        font-family: 'JetBrains Mono', monospace;
        font-size: 3.5rem;
        font-weight: 700;
        color: #FFFFFF !important;
        text-align: center;
        padding: 1rem 0;
    }
    
    .footer-box {
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 0.9rem;
        color: #FFFFFF !important;
        background: rgba(15, 15, 20, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-left: 4px solid #4a90a4;
        padding: 1rem 1.25rem;
        margin-top: 2rem;
        border-radius: 0 12px 12px 0;
        backdrop-filter: blur(14px);
    }
    
    [data-testid="stSlider"] label, [data-testid="stSlider"] span,
    .stSlider label, .stSlider span, [data-testid="stWidgetLabel"] {
        color: #FFFFFF !important;
    }
    
    .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stMarkdown label {
        color: #FFFFFF !important;
    }
    
    .main .block-container { background-color: #1a1a1a !important; padding-top: 1rem; }
    
    div[data-testid="stDataFrame"] {
        background: rgba(15, 15, 20, 0.5);
        border-radius: 12px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown(
    '<p class="main-header">Quantum Telomere Simulation</p>'
    '<p class="subtitle">Modeling biological age uncertainty via quantum superposition</p>',
    unsafe_allow_html=True,
)

# ---- Layout: sliders left, results right ----
params_col, content_col = st.columns([1, 3])

with params_col:
    st.markdown("### Parameters")
    age = st.slider("Age (years)", min_value=20, max_value=90, value=45, step=1)
    stress = st.slider("Stress Level", min_value=0, max_value=100, value=50, step=1)
    st.markdown("#### Research inputs")
    strength_mins = st.slider("Weekly Strength Training (mins)", min_value=0, max_value=300, value=60, step=5)
    body_fat_change = st.slider("Body Fat % Change", min_value=-20.0, max_value=30.0, value=0.0, step=0.5)
    diet_quality = st.slider("Diet Quality", min_value=0, max_value=100, value=60, step=5)
    muscle_score = st.slider("Muscle Score", min_value=0, max_value=100, value=60, step=5)

st.sidebar.caption("Quantum Mode: Multi-Qubit Register Active")

# ---- Load studies table (for display) ----
df_studies = None
try:
    df_studies = pd.read_csv(DATA_PATH)
except Exception:
    df_studies = pd.DataFrame(columns=["Factor", "Effect_Size", "Metric", "Source"])

# ---- Run research-backed quantum simulation ----
# Maps: exercise_mins=strength_mins, weight_gain=body_fat_change
sim_result = run_quantum_simulation(age, stress, strength_mins, body_fat_change, diet_quality, muscle_score)
theta = sim_result["theta"]
prob_0 = sim_result["prob_0"]
prob_1 = sim_result["prob_1"]
entropy_level = sim_result["entropy_level"]
tipping_point_reached = sim_result["tipping_point_reached"]
blue_zone_active = sim_result["blue_zone_active"]
metabolic_milestone = sim_result["metabolic_milestone"]
bloch_coords = sim_result["bloch_coords"]
decay_probability = prob_1
biological_age = sim_result["biological_age"]

# ---- Main content: big Biological Age + chart ----
with content_col:
    # Big Biological Age readout
    st.markdown(
        f'<div class="glass-card">'
        f'<div class="metric-label">Biological Age</div>'
        f'<div class="bio-age-readout">{biological_age} years</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Biological Entropy: {entropy_level:.2%} "
        "(Decoherence Layer from stochastic DNA transcription errors and epigenetic information loss)."
    )
    if tipping_point_reached:
        st.warning("CRITICAL THRESHOLD DETECTED: Epigenetic Resilience Compromised.")
    if metabolic_milestone:
        st.info(
            "RESEARCH NOTE: You are currently in a high-velocity telomere attrition window "
            "(Age 40/70 Milestone)."
        )
    if blue_zone_active:
        st.markdown(
            '<div style="background:#d4af37;color:#1a1a1a;padding:0.6rem 0.9rem;'
            'border-radius:10px;font-weight:700;">'
            "BLUE ZONE SYNERGY ACTIVATED: Enhanced Cellular Protection Enabled."
            "</div>",
            unsafe_allow_html=True,
        )
    st.markdown("")
    chart_col, bloch_col = st.columns(2)

    with chart_col:
        # Plotly: Quantum probability
        st.markdown("#### Quantum Probability of Cellular Decay")
        df_chart = pd.DataFrame({
            "State": ["|0> (resilience)", "|1> (decay)"],
            "Probability": [prob_0, prob_1],
        })
        fig = px.bar(
            df_chart,
            x="State",
            y="Probability",
            color="Probability",
            color_continuous_scale=["#4a90a4", "#e07a5f"],
            text_auto=".2%",
        )
        fig.update_layout(
            xaxis_title="Measurement outcome",
            yaxis_title="Probability",
            yaxis_tickformat=".0%",
            yaxis_range=[0, 1.05],
            showlegend=False,
            font=dict(family="Source Serif 4, Georgia, serif", color="#FFFFFF"),
            margin=dict(t=24, b=48),
            plot_bgcolor="rgba(26,26,26,0.6)",
            paper_bgcolor="rgba(26,26,26,0.6)",
            xaxis=dict(tickfont=dict(color="#FFFFFF"), title_font=dict(color="#FFFFFF")),
            yaxis=dict(tickfont=dict(color="#FFFFFF"), title_font=dict(color="#FFFFFF")),
        )
        fig.update_traces(textposition="outside", textfont=dict(color="#FFFFFF"))
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>')

    with bloch_col:
        st.markdown("#### Genetics Qubit Bloch Sphere")
        bloch_fig = display_bloch_sphere(bloch_coords)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.pyplot(bloch_fig)
        st.markdown('</div>')
        plt.close(bloch_fig)
        st.caption(
            "Statevector of the Genetics Qubit: North Pole (|0>) is the state of maximum biological resilience "
            "and South Pole (|1>) is the state of terminal cellular decay."
        )

    # Data table: research studies
    st.markdown("#### Research Studies")
    if df_studies is not None and len(df_studies) > 0:
        st.dataframe(df_studies, use_container_width=True, hide_index=True)
    else:
        st.caption("No study data loaded. Add study_data.csv with columns: Factor, Effect_Size, Metric, Source.")

    # Footer
    st.markdown(
        """
        <div class="footer-box">
            <strong>About this model:</strong> This simulation uses quantum superposition to model biological uncertainty.
            A 4-qubit register is prepared with RY rotations for Genetics (Age), Exercise, Nutrition, and Adiposity, with entanglement links between exercise-genetics and diet-adiposity.
            Measuring the genetics qubit gives the quantum probability of cellular decay; Biological Age combines this with the research inputs.
            Blue Zone Synergy models Epigenetic Resilience: combined healthy behaviors create a protective buffer against chronological aging.
        </div>
        """,
        unsafe_allow_html=True,
    )
