import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import plotly.express as px

from workflow_features import (
    extract_workflow_features,
    WORKFLOW_FEATURE_NAMES,
    PLATFORM_FEATURE_NAMES,
    PLATFORM_CONFIGS,
)

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

st.set_page_config(
    page_title="Workflow Energy Predictor",
    page_icon="⚡",
    layout="wide",
)

FEATURE_COLS = [
    "ACTUAL_NUM_TASKS", "NUM_EDGES", "DAG_DEPTH", "MAX_PARALLEL_WIDTH",
    "CRITICAL_PATH_TASKS", "NUM_TASK_CATEGORIES", "TOTAL_INPUT_BYTES",
    "PLATFORM_NUM_HOSTS", "PLATFORM_CORES_PER_HOST", "PLATFORM_HOST_SPEED_GF",
    "PLATFORM_NET_BW_MBPS", "PLATFORM_NET_LATENCY_MS", "PLATFORM_DISK_BW_MBPS",
    "PLATFORM_WATT_IDLE", "PLATFORM_WATT_BUSY", "PLATFORM_TOTAL_CORES",
]


@st.cache_resource
def load_models():
    with open(os.path.join(MODEL_DIR, "energy_model.pkl"), "rb") as f:
        model_energy = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "makespan_model.pkl"), "rb") as f:
        model_makespan = pickle.load(f)
    return model_energy, model_makespan


def predict_for_platforms(wf_features, model_energy, model_makespan):
    results = []

    for pcfg in PLATFORM_CONFIGS:
        row = {}
        for c in WORKFLOW_FEATURE_NAMES:
            row[c] = wf_features.get(c, 0)
        for c in PLATFORM_FEATURE_NAMES:
            row[c] = pcfg.get(c, 0)

        X = pd.DataFrame([row])[FEATURE_COLS]
        pred_energy = float(model_energy.predict(X)[0])
        pred_makespan = float(model_makespan.predict(X)[0])

        results.append({
            "Platform": pcfg["name"],
            "Hosts": pcfg["PLATFORM_NUM_HOSTS"],
            "Cores/Host": pcfg["PLATFORM_CORES_PER_HOST"],
            "Speed (Gf)": pcfg["PLATFORM_HOST_SPEED_GF"],
            "Net BW (Mbps)": pcfg["PLATFORM_NET_BW_MBPS"],
            "P_idle (W)": pcfg["PLATFORM_WATT_IDLE"],
            "P_busy (W)": pcfg["PLATFORM_WATT_BUSY"],
            "Predicted Makespan (s)": round(max(pred_makespan, 0), 2),
            "Predicted Energy (J)": round(max(pred_energy, 0), 2),
            "Predicted Energy (Wh)": round(max(pred_energy, 0) / 3_600, 4),
        })

    return pd.DataFrame(results)


def format_bytes(b):
    if b >= 1e9:
        return f"{b/1e9:.1f} GB"
    if b >= 1e6:
        return f"{b/1e6:.1f} MB"
    if b >= 1e3:
        return f"{b/1e3:.1f} KB"
    return f"{b:.0f} B"


def format_time(s):
    if s >= 3600:
        return f"{s/3600:.1f} hours"
    if s >= 60:
        return f"{s/60:.1f} min"
    return f"{s:.2f} sec"


model_energy, model_makespan = load_models()

st.title("Workflow Energy Predictor & Resource Advisor")
st.markdown("Upload a WfFormat JSON workflow file to predict energy consumption and get platform recommendations.")

uploaded = st.file_uploader("Upload workflow JSON", type=["json"])

if uploaded is not None:
    try:
        wf_json = json.load(uploaded)
    except json.JSONDecodeError:
        st.error("Invalid JSON file.")
        st.stop()

    wf_features = extract_workflow_features(wf_json)

    st.header("Workflow Summary")

    with st.container(horizontal=True):
        st.metric("Tasks", wf_features["ACTUAL_NUM_TASKS"], border=True)
        st.metric("Edges", wf_features["NUM_EDGES"], border=True)
        st.metric("DAG Depth", wf_features["DAG_DEPTH"], border=True)
        st.metric("Max Parallel Width", wf_features["MAX_PARALLEL_WIDTH"], border=True)

    with st.container(horizontal=True):
        st.metric("Task Categories", wf_features["NUM_TASK_CATEGORIES"], border=True)
        st.metric("Critical Path Tasks", wf_features["CRITICAL_PATH_TASKS"], border=True)
        st.metric("Total Data", format_bytes(wf_features["TOTAL_INPUT_BYTES"]), border=True)

    st.header("Energy Predictions per Platform")

    results_df = predict_for_platforms(wf_features, model_energy, model_makespan)

    best_idx = results_df["Predicted Energy (J)"].idxmin()
    worst_idx = results_df["Predicted Energy (J)"].idxmax()
    best_platform = results_df.loc[best_idx, "Platform"]
    worst_platform = results_df.loc[worst_idx, "Platform"]

    st.dataframe(
        results_df.style
            .highlight_min(subset=["Predicted Energy (J)"], color="#d1e7dd")
            .highlight_max(subset=["Predicted Energy (J)"], color="#f8d7da"),
        use_container_width=True,
        hide_index=True,
    )

    st.caption("Green = most efficient | Red = least efficient (baseline)")

    st.header("Recommendation")

    best_row = results_df.loc[best_idx]
    worst_energy = results_df.loc[worst_idx, "Predicted Energy (J)"]
    avg_energy = results_df["Predicted Energy (J)"].mean()
    savings_vs_worst = (1 - best_row["Predicted Energy (J)"] / worst_energy) * 100 if worst_energy > 0 else 0
    savings_vs_random = (1 - best_row["Predicted Energy (J)"] / avg_energy) * 100 if avg_energy > 0 else 0

    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Most efficient: {best_platform}**")
        st.error(f"**Least efficient (baseline): {worst_platform}**")
        st.metric("Predicted Makespan", format_time(best_row["Predicted Makespan (s)"]), border=True)
        st.metric("Predicted Energy", f"{best_row['Predicted Energy (Wh)']:.4f} Wh", border=True)
        st.metric(
            f"Savings vs random selection",
            f"{savings_vs_random:.1f}%",
            help="Energy reduction compared to randomly picking a platform (average of all 5)",
            border=True,
        )
        st.metric(
            f"Savings vs {worst_platform}",
            f"{savings_vs_worst:.1f}%",
            help=f"Energy reduction compared to the least efficient platform ({worst_platform})",
            border=True,
        )

    with col2:
        fig = px.bar(
            results_df,
            x="Predicted Energy (J)",
            y="Platform",
            orientation="h",
            color="Platform",
            title="Energy Consumption per Platform",
        )
        fig.update_layout(showlegend=True, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("All extracted features"):
        st.json(wf_features)

else:
    st.info("Upload a WfFormat JSON file to get started. You can generate one with WfCommons.")

    with st.expander("How it works"):
        st.markdown("""
1. **Upload** a WfFormat JSON file (from WfCommons or any WMS)
2. **Feature extraction**: 7 structural workflow features are computed from the DAG (pre-execution only)
3. **Prediction**: Trained ML models predict execution time and energy for each of 5 platform configurations
4. **Recommendation**: The platform with the lowest energy consumption is highlighted

**Platforms**: Based on real servers from SPECpower_ssj2008 benchmark (Q3-Q4 2025) with measured idle/peak wattage.

**Models trained on**: 3,890 WRENCH/SimGrid simulations across 10 workflow types and 5 platforms.

**Accuracy**: Energy R²=0.995, Makespan R²=0.916, Median error 1.7%

**Model details**: See `model_training.ipynb` for full model inspection (features, importances, hyperparameters).
        """)
