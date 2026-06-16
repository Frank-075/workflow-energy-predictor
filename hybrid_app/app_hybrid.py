"""
Interactive hybrid (gray-box) energy predictor and resource advisor.

Mirrors the original Workflow Energy Predictor app (per-platform comparison
table + recommendation), but uses the hybrid model from the thesis. Instead of
predicting energy directly, it learns two platform-aware quantities and
recombines them with the analytical SimGrid power model:

    makespan  T   <- Random Forest on workflow metadata + platform   (transfers across hardware)
    work / FLOPs  <- Random Forest on workflow metadata only          (platform-independent)

    E = num_hosts * watt_idle * T
        + (watt_busy - watt_idle) * (work / (speed_gf * 1e9)) / cores_per_host

Because the power terms (watt_idle, watt_busy, speed, cores) are supplied
explicitly rather than learned, the estimate generalizes to hardware that was
never part of the training set. The 5 known SPECpower platforms are always
compared; you can add your own (unseen) hardware types with the form on the
right and they are ranked alongside the known platforms.
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(BASE, "..")
sys.path.insert(0, os.path.join(REPO, "app"))
from workflow_features import extract_workflow_features  # noqa: E402

DATA_CSV = os.path.join(REPO, "hybrid", "full_dataset.csv")
GRAPHS_JSONL = os.path.join(REPO, "hybrid", "graphs_full.jsonl")
REF = 100e6  # FLOPs per unit of per-task runtime (see thesis dataset generation)

# Structural (pre-execution) workflow features, in the order the models expect.
STRUCT = ["actual_num_tasks", "num_edges", "dag_depth", "max_parallel_width",
          "critical_path_tasks", "num_task_categories", "total_input_bytes"]
# Platform features used by the makespan model.
PLAT = ["num_hosts", "cores_per_host", "total_cores", "speed_gf", "net_bw",
        "net_lat", "disk_bw", "watt_idle", "watt_busy"]

# Map the UPPERCASE keys returned by extract_workflow_features to STRUCT names.
STRUCT_FROM_FEATURES = {
    "actual_num_tasks": "ACTUAL_NUM_TASKS",
    "num_edges": "NUM_EDGES",
    "dag_depth": "DAG_DEPTH",
    "max_parallel_width": "MAX_PARALLEL_WIDTH",
    "critical_path_tasks": "CRITICAL_PATH_TASKS",
    "num_task_categories": "NUM_TASK_CATEGORIES",
    "total_input_bytes": "TOTAL_INPUT_BYTES",
}

# Baseline network/disk parameters of the training data (defaults for the form).
NET_BW, NET_LAT, DISK_BW = 3125.0, 0.01, 1000.0
# I/O-time term: makespan is corrected with an analytical data-movement time.
# alpha was calibrated on the ST45 sensitivity sweeps (alpha ~= 1.0, i.e. the
# physical time = data_volume / bandwidth holds almost exactly for data-bound
# workflows); the compute floor handles the high-bandwidth saturation.
ALPHA_IO = 1.0
BASE_DISK, BASE_NET = 1000.0, 3125.0  # MB/s, the (constant) training bandwidths


@st.cache_resource(show_spinner="Training the hybrid model (one-time, on load) ...")
def load_models():
    """Train the makespan and work models once and cache them for the session."""
    df = pd.read_csv(DATA_CSV)
    graphs = {}
    with open(GRAPHS_JSONL) as fh:
        for line in fh:
            rec = json.loads(line)
            graphs[rec["row_id"]] = rec["graph"]
    df = df[(df.energy_joules > 0) & (df.makespan_sec > 0)].reset_index(drop=True)
    df["work_flops"] = df.row_id.map(lambda r: sum(graphs[r]["runtimes"]) * REF)

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    type_oh = enc.fit_transform(df[["workflow_type"]])

    rf_ms = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf_ms.fit(np.hstack([df[STRUCT + PLAT].to_numpy(), type_oh]),
              np.log10(df["makespan_sec"]))

    rf_w = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf_w.fit(np.hstack([df[STRUCT].to_numpy(), type_oh]),
             np.log10(df["work_flops"]))

    # The 5 known platforms, taken exactly from the training data.
    known = df.drop_duplicates("platform_name")[["platform_name"] + PLAT].to_dict("records")
    return rf_ms, rf_w, enc, sorted(df.workflow_type.unique()), known


def predict_energy(rf_ms, rf_w, enc, struct_vals, wf_type, plat):
    """Run the gray-box pipeline for one workflow + platform spec.

    The Random Forest learned makespan at the training bandwidths (disk 1000,
    net 3125 MB/s), where those features had zero variance. To make disk/network
    bandwidth influence the estimate, the makespan is corrected with a physical
    data-movement term T_io = total_input_bytes / (min(disk, net) * 1e6), floored
    by the compute time T_cpu so it cannot drop below what the cores can do.
    """
    type_oh = enc.transform(pd.DataFrame({"workflow_type": [wf_type]}))
    struct = np.array([[struct_vals[c] for c in STRUCT]], dtype=float)
    plat_vec = np.array([[plat[c] for c in PLAT]], dtype=float)

    t_rf = float(10 ** rf_ms.predict(np.hstack([struct, plat_vec, type_oh]))[0])
    work = float(10 ** rf_w.predict(np.hstack([struct, type_oh]))[0])

    # Compute floor: busy-core-seconds spread over the available cores.
    t_cpu = work / (plat["speed_gf"] * 1e9 * plat["cores_per_host"])
    # Analytical data-movement time through the slowest link (bytes -> MB -> s).
    bytes_total = float(struct_vals["total_input_bytes"])
    b_eff = min(plat["disk_bw"], plat["net_bw"])
    t_io = bytes_total / (b_eff * 1e6)
    t_io_base = bytes_total / (min(BASE_DISK, BASE_NET) * 1e6)
    makespan = max(t_cpu, t_rf + ALPHA_IO * (t_io - t_io_base))

    busy_core_sec = work / (plat["speed_gf"] * 1e9)
    energy = (plat["num_hosts"] * plat["watt_idle"] * makespan
              + (plat["watt_busy"] - plat["watt_idle"]) * busy_core_sec / plat["cores_per_host"])
    return energy, makespan, work


def predict_all(rf_ms, rf_w, enc, struct_vals, wf_type, platforms):
    rows = []
    for plat in platforms:
        energy, makespan, work = predict_energy(rf_ms, rf_w, enc, struct_vals, wf_type, plat)
        rows.append({
            "Platform": plat["name"],
            "In training?": "yes" if plat.get("seen") else "no (unseen)",
            "Hosts": int(plat["num_hosts"]),
            "Cores/Host": int(plat["cores_per_host"]),
            "Speed (Gf)": plat["speed_gf"],
            "P_idle (W)": plat["watt_idle"],
            "P_busy (W)": plat["watt_busy"],
            "Predicted Makespan (s)": round(max(makespan, 0), 2),
            "Predicted Energy (J)": round(max(energy, 0), 2),
            "Predicted Energy (Wh)": round(max(energy, 0) / 3_600, 4),
        })
    return pd.DataFrame(rows)


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


# --------------------------------------------------------------------------- UI
st.set_page_config(page_title="Hybrid Energy Predictor", page_icon="⚡", layout="wide")
st.title("Hybrid (gray-box) Workflow Energy Predictor & Resource Advisor")
st.markdown(
    "Upload a WfFormat JSON workflow to predict energy and makespan and get a "
    "platform recommendation. Unlike the direct model, this gray-box model "
    "reconstructs energy from a learned makespan and compute-work plus the "
    "analytical SimGrid power model, so you can also evaluate **your own "
    "hardware that was never part of the training set**. Disk and network "
    "bandwidth feed into a physical data-movement term, so changing them for an "
    "unseen server changes the predicted makespan and energy."
)

rf_ms, rf_w, enc, known_types, known_platforms = load_models()

if "custom_platforms" not in st.session_state:
    st.session_state.custom_platforms = []

left, right = st.columns(2)

with left:
    st.subheader("1. Workflow")
    uploaded = st.file_uploader("Upload a WfFormat workflow JSON", type=["json"])
    struct_vals, detected_type = None, None
    if uploaded is not None:
        try:
            wf_json = json.load(uploaded)
        except Exception as exc:
            st.error(f"Could not parse JSON: {exc}")
            wf_json = None
        if wf_json is not None:
            raw = extract_workflow_features(wf_json)
            struct_vals = {k: raw[STRUCT_FROM_FEATURES[k]] for k in STRUCT}
            name = wf_json.get("name", "")
            detected_type = name.split("-")[0] if name else ""
            st.success(f"Detected workflow type: **{detected_type or 'unknown'}**")

with right:
    st.subheader("2. Add your own hardware (optional)")
    st.caption("The 5 known SPECpower platforms are always compared. Add unseen platforms here.")
    with st.form("add_platform", clear_on_submit=True):
        name = st.text_input("Platform name", value="my_server")
        c1, c2 = st.columns(2)
        f_hosts = c1.number_input("Number of hosts", min_value=1, value=1, step=1)
        f_cores = c2.number_input("Cores per host", min_value=1, value=64, step=1)
        f_speed = c1.number_input("Core speed (GFlops)", min_value=0.1, value=200.0, step=1.0, format="%.1f")
        f_idle = c2.number_input("Idle power (W)", min_value=0.0, value=60.0, step=1.0, format="%.1f")
        f_busy = c1.number_input("Busy power (W)", min_value=0.0, value=300.0, step=1.0, format="%.1f")
        f_diskbw = c2.number_input("Disk bandwidth (MB/s)", min_value=1.0, value=DISK_BW, step=50.0)
        f_netbw = c1.number_input("Network bandwidth (MB/s)", min_value=1.0, value=NET_BW, step=50.0)
        st.caption("Disk and network bandwidth affect the estimate through the "
                   "analytical data-movement term (lower bandwidth -> longer "
                   "makespan -> higher energy). Training baseline: 1000 / 3125 MB/s.")
        with st.expander("Advanced (network latency, minor effect)"):
            f_netlat = st.number_input("Network latency (ms)", min_value=0.0, value=NET_LAT, step=0.01, format="%.2f")
        submitted = st.form_submit_button("Add platform", type="primary")
        if submitted:
            if not name.strip():
                st.warning("Please give the platform a name.")
            else:
                st.session_state.custom_platforms.append({
                    "name": name.strip(), "seen": False,
                    "num_hosts": int(f_hosts), "cores_per_host": int(f_cores),
                    "total_cores": int(f_hosts) * int(f_cores), "speed_gf": float(f_speed),
                    "net_bw": float(f_netbw), "net_lat": float(f_netlat), "disk_bw": float(f_diskbw),
                    "watt_idle": float(f_idle), "watt_busy": float(f_busy),
                })

    if st.session_state.custom_platforms:
        st.markdown("**Added platforms:** " +
                    ", ".join(p["name"] for p in st.session_state.custom_platforms))
        if st.button("Clear added platforms"):
            st.session_state.custom_platforms = []
            st.rerun()

st.divider()

if struct_vals is None:
    st.info("Upload a WfFormat JSON file to get started. You can generate one with WfCommons.")
    with st.expander("How it works"):
        st.markdown("""
1. **Upload** a WfFormat JSON file (from WfCommons or any WMS)
2. **Feature extraction**: 7 structural workflow features are computed from the DAG (pre-execution only)
3. **Gray-box prediction**: a Random Forest predicts makespan and another predicts compute work; energy is then **reconstructed** with the analytical SimGrid power model using each platform's hardware specs. Disk/network bandwidth enter through a physical data-movement term (time = data / bandwidth), calibrated against the platform sensitivity sweeps
4. **Recommendation**: the platform with the lowest predicted energy is highlighted, including any unseen hardware you added

**Why hybrid?** The direct model cannot extrapolate energy to platforms outside its training set. By learning only the platform-independent quantities (work) and the transferable makespan, and plugging the real hardware specs into the energy formula, this model generalizes to **unseen** platforms.
        """)
else:
    st.header("Workflow Summary")
    with st.container():
        cols = st.columns(4)
        cols[0].metric("Tasks", struct_vals["actual_num_tasks"], border=True)
        cols[1].metric("Edges", struct_vals["num_edges"], border=True)
        cols[2].metric("DAG Depth", struct_vals["dag_depth"], border=True)
        cols[3].metric("Max Parallel Width", struct_vals["max_parallel_width"], border=True)
        cols2 = st.columns(4)
        cols2[0].metric("Task Categories", struct_vals["num_task_categories"], border=True)
        cols2[1].metric("Critical Path Tasks", struct_vals["critical_path_tasks"], border=True)
        cols2[2].metric("Total Data", format_bytes(struct_vals["total_input_bytes"]), border=True)

    type_for_pred = st.selectbox(
        "Workflow type used for the prediction (auto-detected, override if needed)",
        known_types,
        index=known_types.index(detected_type) if detected_type in known_types else 0,
    )

    platforms = [{"name": k["platform_name"], "seen": True, **{c: k[c] for c in PLAT}}
                 for k in known_platforms] + st.session_state.custom_platforms

    results_df = predict_all(rf_ms, rf_w, enc, struct_vals, type_for_pred, platforms)

    st.header("Energy Predictions per Platform")
    best_idx = results_df["Predicted Energy (J)"].idxmin()
    worst_idx = results_df["Predicted Energy (J)"].idxmax()
    best_platform = results_df.loc[best_idx, "Platform"]
    worst_platform = results_df.loc[worst_idx, "Platform"]

    _emin = results_df["Predicted Energy (J)"].min()
    _emax = results_df["Predicted Energy (J)"].max()

    def _style_energy(col):
        out = []
        for v in col:
            if v == _emin:
                out.append("background-color:#d1e7dd; color:#0f5132; font-weight:600")
            elif v == _emax:
                out.append("background-color:#f8d7da; color:#842029; font-weight:600")
            else:
                out.append("")
        return out

    st.dataframe(
        results_df.style.apply(_style_energy, subset=["Predicted Energy (J)"]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Green = most efficient | Red = least efficient (baseline). "
               "'In training? = no' marks hardware unseen during training.")

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
        st.metric("Savings vs random selection", f"{savings_vs_random:.1f}%",
                  help="Energy reduction compared to randomly picking a platform (average of all)",
                  border=True)
        st.metric(f"Savings vs {worst_platform}", f"{savings_vs_worst:.1f}%",
                  help=f"Energy reduction compared to the least efficient platform ({worst_platform})",
                  border=True)
    with col2:
        fig = px.bar(
            results_df.sort_values("Predicted Energy (J)"),
            x="Predicted Energy (J)", y="Platform", orientation="h",
            color="In training?",
            color_discrete_map={"yes": "#1f77b4", "no (unseen)": "#ff7f0e"},
            title="Predicted Energy Consumption per Platform",
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Energy is reconstructed analytically: "
        "E = num_hosts x watt_idle x makespan + (watt_busy - watt_idle) x work / (speed x 1e9) / cores_per_host. "
        "Makespan is corrected with a data-movement term: "
        "makespan = max(compute_time, RF_makespan + total_bytes / (min(disk, net) x 1e6) - baseline_io)."
    )

    with st.expander("All extracted features"):
        st.json(struct_vals)
