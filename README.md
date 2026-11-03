# 🧠⚔️ PREDATOR NET

> **Adaptive Neural Network for Cybersecurity Intrusion Detection Through Knowledge-Preserving Neuron Absorption**  
> Presentation Tagline: *“A cybersecurity neural network that learns which neurons it needs — and eliminates the rest.”*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 69 Passed](https://img.shields.io/badge/tests-69%20passed-brightgreen.svg)](tests/)

---

## 🎯 Research Objective

Can an artificial neural network reduce its structural complexity through **knowledge-preserving neuron absorption** while maintaining intrusion-detection performance?

PREDATOR NET addresses the challenge of overparameterization in Network Intrusion Detection Systems (NIDS). Rather than simply zeroing weights (masking) or discarding parameters without compensation (conventional pruning), PREDATOR NET:
1. Calculates measurable first-order behavioral importance for each hidden neuron.
2. Identifies low-importance weak neurons and finds compatible strong neurons using activation & weight cosine similarities.
3. Transfers learned representation weights into compatible strong neurons.
4. **Physically reconstructs the adjacent linear layers** to achieve real parameter reduction.
5. Employs a validation gate with automatic rollback to ensure intrusion detection performance is strictly maintained.

---

## 🔬 Core Research Benchmark (Spec §41)

All 6 models evaluated on identical test flow traffic with zero data leakage:

| Model | Neurons | Parameters | Accuracy | Precision | Recall (Det Rate) | F1 Score | PR-AUC | FPR | Throughput (samples/s) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Dense ANN (Baseline)** | 224 | 13,825 | 0.9987 | 0.9934 | 1.0000 | 0.9967 | 1.0000 | 0.0017 | 3,880,228 |
| **PREDATOR-Light** | 223 | 13,759 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 3,472,340 |
| **Random Pruning** | 223 | 13,759 | 0.9987 | 0.9934 | 1.0000 | 0.9967 | 1.0000 | 0.0017 | 5,324,515 |
| **Structured Pruning** | 223 | 13,759 | 0.9987 | 0.9934 | 1.0000 | 0.9967 | 1.0000 | 0.0017 | 5,150,802 |
| **PREDATOR-Full** | 223 | 13,759 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 4,103,452 |
| **PREDATOR-D** | 223 | 13,759 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 3,622,881 |

> **Note:** Values above are strictly populated from actual experimental measurements stored in `artifacts/evaluation/core_research_table.csv`.

---

## ⚙️ Architecture & PREDATOR Pipeline

```
  [Input Features] (26 behavioral flow features)
         │
  [Layer 1: Linear(26, 128) + ReLU]
         │
  [Layer 2: Linear(128, 64) + ReLU]
         │
  [Layer 3: Linear(64, 32)  + ReLU]
         │
  [Output:  Linear(32, 1)   → Logit]
```

### The 5-Step Absorption Cycle:
1. **First-Order Importance:** $\text{importance}_j = \mathbb{E}\left[|a_j \cdot \frac{\partial L}{\partial a_j}|\right]$ combined with outgoing weight norm $\|U_j\|_1$.
2. **Compatibility Selection:** $\text{compatibility}(w, s) = 0.70 \cdot \cos(H_w, H_s) + 0.30 \cdot \cos(V_w, V_s)$.
   - Gated: requires compatibility $\ge 0.70$ (rejects otherwise).
3. **Absorption Operator:**
   - **PREDATOR-LIGHT:** $U_s^{\text{new}} = U_s + \alpha U_w$ where $\alpha = \text{clip}(\text{compatibility} \cdot \frac{\text{imp}_w}{\text{imp}_s + \epsilon}, 0.05, 0.50)$.
   - **PREDATOR-FULL:** Also absorbs incoming weights $V_s^{\text{new}} = (1-\gamma)V_s + \gamma V_w$ and bias.
4. **Structural Rebuild:** Row $w$ removed from Layer $L$, column $w$ removed from Layer $L+1$. Sliced into brand new `nn.Linear` layers.
5. **Validation Gate:** Accepted only if $\Delta F_1 \le 0.005$ and $\Delta \text{Recall} \le 0.005$; otherwise **exact rollback** is performed.

---

## 📁 Repository Structure

```
Predator-NET---Ann-Project/
├── config/
│   ├── baseline.yaml            # Baseline ANN & data hyperparameters
│   ├── predator.yaml            # PREDATOR absorption thresholds & policies
│   └── experiments.yaml         # Ablation & benchmark experiment settings
├── src/predator_net/
│   ├── data/
│   │   ├── loader.py            # CIC-IDS2017 multi-file loader & column normalizer
│   │   ├── cleaner.py           # Missing/Inf/duplicate handling & label encoder
│   │   ├── splitter.py          # Protocol A (stratified) & B (leakage-aware) splits
│   │   ├── scaler.py            # FeatureScaler enforcing fit-on-train semantics
│   │   ├── dataset.py           # High-performance PyTorch FlowDataset & DataLoader
│   │   └── synthetic.py         # Realistic synthetic CIC-IDS2017 flow generator
│   ├── features/
│   │   ├── engineering.py       # Mode A (Behavioral) & Mode B (Enriched) extraction
│   │   ├── selection.py         # Collinearity filtering (0.95) & Mutual Information
│   │   └── analysis.py          # Rankings exporter & correlation heatmaps
│   ├── models/
│   │   ├── baseline_mlp.py      # Standard reference Dense ANN
│   │   ├── dynamic_mlp.py       # Dynamic nn.ModuleList MLP supporting live rebuild
│   │   └── model_utils.py       # Parameter counting & numerical integrity checks
│   ├── predator/
│   │   ├── importance.py        # First-order contribution & weight scoring
│   │   ├── similarity.py        # Activation profile & weight cosine compatibility
│   │   ├── absorption.py        # PREDATOR-Light and PREDATOR-Full operators
│   │   ├── structural_rebuild.py# Physical Linear layer slicing and dimension update
│   │   ├── distillation.py      # Knowledge distillation loss for PREDATOR-D
│   │   └── controller.py        # Iterative loop, validation gate, rollback & logging
│   ├── evaluation/
│   │   ├── metrics.py           # F1, Recall, Precision, FPR, FNR, PR-AUC, ROC-AUC
│   │   ├── latency.py           # Single-sample latency & throughput benchmarking
│   │   ├── benchmark.py         # 6-model comparison orchestrator
│   │   └── ablation.py          # Feature ablation & noise robustness analysis
│   ├── visualization/
│   │   ├── plots.py             # Confusion matrix, ROC, PR, training curves
│   │   └── evolution.py         # 8 publication graphs from Spec §49 & §50
│   └── demo/
│       ├── replay.py            # Terminal SOC live flow monitor
│       └── dashboard.py         # Interactive Streamlit SOC web dashboard
├── scripts/
│   ├── prepare_dataset.py       # Phase 1: Ingestion & quality report
│   ├── preprocess_data.py       # Phase 2: Feature selection, split, scale, save
│   ├── train_baseline.py        # Phase 3: Train baseline reference ANN
│   ├── run_predator.py          # Phase 8: Execute iterative absorption loop
│   ├── run_benchmarks.py        # Phase 9: Execute 6-model benchmark suite
│   ├── run_ablations.py         # Phase 10: Run ablations & generate 8 plots
│   ├── run_demo.py              # Phase 11: Launch CLI replay or Streamlit UI
│   └── generate_report.py       # Phase 12: Generate technical paper report
├── tests/                       # 69 Unit Tests (100% passing)
└── artifacts/                   # Verified experimental artifacts & figures
```

---

## 🚀 Quickstart & Pipeline Execution

### 1. Installation

```bash
git clone https://github.com/Zacker/Predator-NET---Ann-Project.git
cd Predator-NET---Ann-Project
pip install -e ".[dev]"
```

### 2. Run Test Suite

Verify all 69 unit tests passing:
```bash
pytest tests/ -v
```

### 3. Step-by-Step Phase Execution

```bash
# Phase 1 & 2: Dataset Preprocessing & Feature Selection
# (Uses CIC-IDS2017 CSVs in data/raw or generates synthetic validation data)
python scripts/preprocess_data.py --generate-synthetic

# Phase 3: Train Dense ANN Baseline
python scripts/train_baseline.py

# Phase 8: Run PREDATOR Neuron Absorption Loop
python scripts/run_predator.py

# Phase 9: Run 6-Model Comparative Benchmark Suite
python scripts/run_benchmarks.py

# Phase 10: Run Feature Ablation, Robustness & Generate 8 Publication Graphs
python scripts/run_ablations.py

# Phase 11: Launch Live SOC Demo (Terminal or Streamlit Dashboard)
python scripts/run_demo.py --mode terminal --flows 10
# OR: streamlit run src/predator_net/demo/dashboard.py

# Phase 12: Generate Research Technical Report
python scripts/generate_report.py
```

### 4. 🖥️ Native Desktop Cockpit (Electron + React 19 + AI Engine)

PREDATOR-NET includes an interactive real-time Electron desktop application featuring GPU-accelerated 2D neural network activation simulation, live packet interception across browser and OS traffic, and dynamic firewall controls:

```bash
# 1. Install frontend dependencies (first time only)
cd desktop-app
npm install
cd ..

# 2. Launch Desktop Cockpit (Starts AI Engine :8765 + Target Server :8080 + Native Desktop Window)
python launch_cockpit.py
# Or on Windows, double-click launch_cockpit.bat
```

### 5. ⚔️ Live Cross-Machine Attack Demonstration (k6 Flood)

Demonstrate live volumetric DDoS intrusion detection and manual firewall blocking across two laptops on the same Wi-Fi / LAN:

1. **Host Laptop (PREDATOR-NET):** Launch the cockpit (`python launch_cockpit.py`). Note the host Wi-Fi LAN IP displayed in the GUI header (e.g. `10.5.128.179`).
2. **Attacker Laptop (Secondary Machine):** Run the prepared k6 attack script:
   ```bash
   k6 run k6_attack_script.js
   ```
3. **Detection & Real-Time Alert:** PREDATOR ANN detects high request velocity (`reqs_per_sec > 15`), raises the threat score to **98.5%**, and displays an active intrusion alert banner displaying the attacker's IP.
4. **Manual Firewall Mitigation:** Click **`[ Block IP in Firewall ]`** in the dashboard. PREDATOR-NET immediately blacklists the attacker IP and responds to subsequent requests with `403 Forbidden`, causing k6 checks on the attacker laptop to fail with 100% rejection.
5. **Malicious Web Detection:** Real-time socket sniffer inspects browser connections. Visiting test threat sites (`testsafebrowsing.appspot.com`, `malware.wicar.org`, `eicar.org`) or clicking **`[ Harmful Site Test ]`** in the cockpit immediately raises a live intrusion event.

---

## 📊 Feature Ablation Insights (Spec §47)

| Feature Group Removed | Features Dropped | Test Recall | Test F1 Score | F1 Degradation |
|:---|---:|---:|---:|---:|
| **All Features (Reference)** | 0 | 1.0000 | 0.9967 | 0.0000 |
| **Without Ports** | 1 | 1.0000 | 0.9967 | 0.0000 |
| **Without TCP Flags** | 18 | 0.6600 | 0.7734 | **-0.2232** |
| **Without Timing Attributes** | 3 | 1.0000 | 1.0000 | -0.0033 |
| **Without Packet Statistics** | 7 | 1.0000 | 0.9868 | -0.0098 |
| **Without Byte Statistics** | 3 | 1.0000 | 0.9934 | -0.0033 |

> **Key Cybersecurity Finding:** TCP flag features (SYN, ACK, RST, FIN) are the primary detection signal. Removing TCP flags drops model F1 score by 22.3%, confirming the model learns behavioral protocol mechanics.

---

## 🛡️ Critical Rules Enforced (Spec §70)

1. **Zero Hallucination / Zero Fabrication:** All metric cells originate strictly from executing experimental code.
2. **Strict Fit-on-Train:** Feature scalers are fit exclusively on `X_train` and never touch validation or test sets.
3. **No Zero-Weight Masking:** Structural rebuild physically resizes matrices into smaller `nn.Linear` layers.
4. **Dynamic Feature Count:** Model input dimensions adapt dynamically to selected features.
5. **Deterministic Seeding:** PyTorch, NumPy, and Python seeds are preserved (`seed=42`).

---

## 📜 Citation & License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
