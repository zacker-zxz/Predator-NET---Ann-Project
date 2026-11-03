"""
PREDATOR NET — Interactive Streamlit SOC & Research Dashboard.

Provides an executive interface for:
  1. Live SOC Flow Replay & Attack Detection
  2. Neural Network Architecture Evolution (Before vs After Absorption)
  3. Spec §41 Core Research Benchmarking Table
  4. Publication Figure Gallery & Feature Ablation Insights
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

st.set_page_config(
    page_title="PREDATOR NET — Adaptive NIDS Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠⚔️ PREDATOR NET")
st.subheader("Adaptive Neural Network for Cybersecurity Intrusion Detection Through Knowledge-Preserving Neuron Absorption")
st.caption("“A cybersecurity neural network that learns which neurons it needs — and eliminates the rest.” (Spec §1)")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select View",
    ["Executive Summary", "Core Research Table (Spec §41)", "Live SOC Replay", "Evolution & Visualizations", "Feature Ablation"]
)

artifacts_dir = PROJECT_ROOT / "artifacts"
eval_dir = artifacts_dir / "evaluation"
figures_dir = artifacts_dir / "figures"
pred_dir = artifacts_dir / "predator"
models_dir = artifacts_dir / "models"
processed_dir = PROJECT_ROOT / "data" / "processed"

# 1. Executive Summary
if page == "Executive Summary":
    st.markdown("### System Architecture & Compression Summary")

    summary_file = pred_dir / "predator_summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            summary = json.load(f)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active Neurons", summary["final_hidden_neurons"], f"-{summary['neuron_reduction_percent']:.1f}%")
        c2.metric("Parameters", f"{summary['final_parameters']['total_parameters']:,}", f"-{summary['parameter_reduction_percent']:.1f}%")
        c3.metric("Test F1 Score", f"{summary['metrics']['f1']:.4f}")
        c4.metric("Throughput", f"{summary['latency_and_throughput']['throughput_samples_per_sec']:,.0f} flows/s")

        st.info(f"**Baseline Architecture:** {' → '.join(map(str, summary['initial_architecture']))}  \n"
                f"**PREDATOR Compressed Architecture:** {' → '.join(map(str, summary['final_architecture']))}")

        st.markdown("#### Test Set Performance Metrics")
        m = summary["metrics"]
        st.dataframe(pd.DataFrame([{
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall (Detection Rate)": f"{m['recall']:.4f}",
            "F1 Score": f"{m['f1']:.4f}",
            "False Positive Rate": f"{m['false_positive_rate']:.4f}",
            "PR-AUC": f"{m['pr_auc']:.4f}",
            "ROC-AUC": f"{m['roc_auc']:.4f}",
        }]), use_container_width=True)
    else:
        st.warning("Run scripts/train_baseline.py and scripts/run_predator.py first to populate summary.")

# 2. Core Research Table
elif page == "Core Research Table (Spec §41)":
    st.markdown("### Spec §41 Core Research Benchmarking Table")
    st.markdown("> Strict Rule: No cell is populated until actual experimental execution yields verified results.")

    table_file = eval_dir / "core_research_table.csv"
    if table_file.exists():
        df_table = pd.read_csv(table_file)
        st.dataframe(df_table, use_container_width=True)
    else:
        st.warning("Run scripts/run_benchmarks.py to generate the 6-model benchmark table.")

# 3. Live SOC Replay
elif page == "Live SOC Replay":
    st.markdown("### Simulated Live SOC Traffic Stream (Spec §51)")
    st.caption("Streams preprocessed CIC-IDS2017 test traffic flows to demonstrate real-time attack inference.")

    max_flows = st.slider("Flows to simulate", min_value=5, max_value=100, value=15)
    speed = st.slider("Simulation interval (seconds)", min_value=0.05, max_value=1.0, value=0.2)

    if st.button("Start Flow Replay"):
        comp_ckpt = pred_dir / "predator_compressed.pt"
        if comp_ckpt.exists() and (processed_dir / "test_X.npy").exists():
            from predator_net.models.dynamic_mlp import DynamicMLP
            ckpt = torch.load(comp_ckpt, map_location="cpu", weights_only=False)
            model = DynamicMLP(input_dim=ckpt["input_dim"], hidden_dims=ckpt["hidden_dims"])
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            X_test = np.load(processed_dir / "test_X.npy")
            y_test = np.load(processed_dir / "test_y.npy")

            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            table_placeholder = st.empty()

            logs = []
            for i in range(min(max_flows, len(X_test))):
                x_ten = torch.from_numpy(X_test[i]).unsqueeze(0)
                with torch.no_grad():
                    prob = float(torch.sigmoid(model(x_ten)).item())

                is_attack = prob >= 0.5
                pred = "⚠️ ATTACK" if is_attack else "✓ BENIGN"
                actual = "ATTACK" if int(y_test[i]) == 1 else "BENIGN"

                logs.append({
                    "Flow #": i + 1,
                    "Prediction": pred,
                    "Confidence": f"{prob*100:.1f}%" if is_attack else f"{(1-prob)*100:.1f}%",
                    "Ground Truth": actual,
                    "Active Neurons": sum(model.hidden_dims),
                })

                table_placeholder.dataframe(pd.DataFrame(logs).iloc[::-1], use_container_width=True)
                progress_bar.progress((i + 1) / max_flows)
                time.sleep(speed)
        else:
            st.error("Processed data or compressed model missing.")

# 4. Evolution & Visualizations
elif page == "Evolution & Visualizations":
    st.markdown("### Spec §49 & §50 Publication Visualizations")
    figs = list(figures_dir.glob("*.png"))
    if figs:
        cols = st.columns(2)
        for i, f_path in enumerate(figs):
            with cols[i % 2]:
                st.image(str(f_path), caption=f_path.stem.replace("_", " ").title())
    else:
        st.warning("Run scripts/run_ablations.py to generate publication figures.")

# 5. Feature Ablation
elif page == "Feature Ablation":
    st.markdown("### Spec §47 Feature Ablation Analysis")
    abl_file = eval_dir / "feature_ablation.csv"
    if abl_file.exists():
        df_abl = pd.read_csv(abl_file)
        st.dataframe(df_abl, use_container_width=True)
        st.bar_chart(df_abl.set_index("Feature Group")["F1"])
    else:
        st.warning("Run scripts/run_ablations.py to generate feature ablation results.")
