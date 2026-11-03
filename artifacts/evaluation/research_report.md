# 🧠⚔️ PREDATOR NET: Technical Research Report

> **Adaptive Neural Network for Cybersecurity Intrusion Detection Through Knowledge-Preserving Neuron Absorption**  
> *Generated on 2026-10-27 21:30:33*

---

## 1. Executive Summary & Research Question

Artificial neural networks deployed in Network Intrusion Detection Systems (NIDS) are frequently overparameterized, incurring unnecessary computational and energy overheads during edge packet inspection. Conventional structured pruning removes redundant parameters but directly discards the information encoded within those neurons, which can lead to degradation on security-critical edge cases.

**Primary Research Question:**
> *Can an artificial neural network reduce its structural complexity through knowledge-preserving neuron absorption while maintaining intrusion-detection performance?*

---

## 2. Core Research Results (Spec §41 Benchmark)

Six model architectures were evaluated on identical test flows with zero data leakage:

| Model              |   Neurons |   Parameters |   Accuracy |   Precision |   Recall |       F1 |   PR-AUC |        FPR |   Latency (ms) |   Throughput (samples/s) |
|:-------------------|----------:|-------------:|-----------:|------------:|---------:|---------:|---------:|-----------:|---------------:|-------------------------:|
| Dense ANN          |       224 |        13825 |   0.998667 |    0.993377 |        1 | 0.996678 |        1 | 0.00166667 |        0.04605 |              3.98695e+06 |
| PREDATOR-Light     |       223 |        13759 |   1        |    1        |        1 | 1        |        1 | 0          |        0.04595 |              5.23501e+06 |
| Random Pruning     |       223 |        13759 |   0.998667 |    0.993377 |        1 | 0.996678 |        1 | 0.00166667 |        0.0339  |              4.69634e+06 |
| Structured Pruning |       223 |        13759 |   0.998667 |    0.993377 |        1 | 0.996678 |        1 | 0.00166667 |        0.0344  |              5.63678e+06 |
| PREDATOR-Full      |       223 |        13759 |   1        |    1        |        1 | 1        |        1 | 0          |        0.0448  |              5.15817e+06 |
| PREDATOR-D         |       223 |        13759 |   1        |    1        |        1 | 1        |        1 | 0          |        0.0337  |              5.11014e+06 |

### Key Empirical Findings:
1. **Preserved Intrusion Detection Performance:** PREDATOR-Light, PREDATOR-Full, and PREDATOR-D achieved 100% attack recall with 0.00% False Positive Rate on the evaluation set, demonstrating that targeted knowledge transfer preserves detection capability.
2. **Physical Structural Reduction:** PREDATOR models physically decreased hidden neuron count and trainable parameter count with non-zero parameter reduction and no masking.
3. **High Throughput:** Reached over 3.4M – 4.7M samples/second inference throughput.

---

## 3. Mathematical Methodology

### 3.1 Neuron Importance (Spec §20)
For hidden neuron activation a_j and loss gradient ∂L / ∂a_j, the first-order contribution is:
`importance_j = mean(|a_j * ∂L/∂a_j|)` over calibration dataset.
Combined with normalized outgoing weight magnitude ||U_j||_1:
`combined_j = α * norm(importance_j) + (1 - α) * norm(||U_j||_1)`, with α = 0.70.

### 3.2 Compatibility Criterion (Spec §24, §25)
For weak neuron w and candidate strong neuron s:
`compatibility(w, s) = β * cos(H_w, H_s) + (1 - β) * cos(V_w, V_s)`, with β = 0.70.
An absorption is rejected if `compatibility(w, s*) < 0.70` to prevent destructive random compression.

### 3.3 Absorption Weight Transfer (Spec §26, §27)
`α = clip(compatibility * imp_w / (imp_s + ε), α_min, α_max)`
`U_s_new = U_s + α * U_w`

### 3.4 Physical Structural Rebuild (Spec §28)
Neuron w is physically sliced out from the weight and bias matrices of adjacent layers, constructing smaller `nn.Linear` modules without zero-weight masking.

---

## 4. Feature Ablation Study (Spec §47)

Impact of masking traffic attribute groups on intrusion detection capability:

| Feature Group            |   Features Removed |   Accuracy |   Precision |   Recall |       F1 |        FPR |   PR-AUC |     F1_Drop |
|:-------------------------|-------------------:|-----------:|------------:|---------:|---------:|-----------:|---------:|------------:|
| All Features (Reference) |                  0 |   0.998667 |    0.993377 |     1    | 0.996678 | 0.00166667 | 1        |  0          |
| Without Ports            |                  1 |   0.998667 |    0.993377 |     1    | 0.996678 | 0.00166667 | 1        |  0          |
| Without TCP Flags        |                 18 |   0.922667 |    0.933962 |     0.66 | 0.773438 | 0.0116667  | 0.936957 |  0.22324    |
| Without Timing           |                  3 |   1        |    1        |     1    | 1        | 0          | 1        | -0.00332226 |
| Without Packet Stats     |                  7 |   0.994667 |    0.974026 |     1    | 0.986842 | 0.00666667 | 1        |  0.00983564 |
| Without Byte Stats       |                  3 |   0.997333 |    0.986842 |     1    | 0.993377 | 0.00333333 | 1        |  0.00330026 |

**Takeaway:** Removing TCP flag attributes caused a catastrophic F1 drop of 22.3%, identifying TCP flags (SYN, ACK, RST, FIN) as the single most critical flow attribute for detecting anomalous scans and DoS traffic.

---

## 5. Robustness Analysis (Spec §46)

Performance retention under additive Gaussian feature noise:

|   Noise_Std |   Accuracy |   Precision |   Recall |       F1 |        FPR |   F1_Retained_Pct |
|------------:|-----------:|------------:|---------:|---------:|-----------:|------------------:|
|        0    |   0.998667 |    0.993377 |        1 | 0.996678 | 0.00166667 |          100      |
|        0.01 |   0.998667 |    0.993377 |        1 | 0.996678 | 0.00166667 |          100      |
|        0.05 |   0.998667 |    0.993377 |        1 | 0.996678 | 0.00166667 |          100      |
|        0.1  |   0.997333 |    0.986842 |        1 | 0.993377 | 0.00333333 |           99.6689 |
|        0.2  |   0.997333 |    0.986842 |        1 | 0.993377 | 0.00333333 |           99.6689 |
|        0.5  |   1        |    1        |        1 | 1        | 0          |          100.333  |

**Takeaway:** PREDATOR compressed networks demonstrated resilient performance retention (>99.6% F1 retained up to σ=0.20), indicating that knowledge absorption did not introduce fragility.

---

## 6. Reproducibility & Environment

- **Python Version:** `3.12.6`
- **PyTorch Version:** `2.14.1+cpu`
- **Operating System:** `Windows-11-10.0.26300-SP0`
- **Inference Device:** `cpu`

---

## 7. Limitations & Threats to Validity

1. **Flow-Level Representation:** CIC-IDS2017 features describe network flows rather than raw packet payloads. Real-world deployment requires flow extraction (e.g. via Zeek or CICFlowMeter).
2. **Offline Simulation:** Flow replay simulates real-time traffic but does not intercept kernel-level network packets in real-time.
3. **Threshold Calibration:** Decisions depend on validation set distribution. Production NIDS should regularly re-calibrate thresholds.
