# PREDATOR NET
## Adaptive Neural Network for Cybersecurity Intrusion Detection Through Knowledge-Preserving Neuron Absorption

---

# 1. PROJECT IDENTITY

Project name:

**PREDATOR NET**

Academic name:

**PREDATOR NET: An Adaptive Structured Neural Network for Network Intrusion Detection Using Knowledge-Preserving Neuron Absorption**

Presentation tagline:

**“A cybersecurity neural network that learns which neurons it needs — and eliminates the rest.”**

Core idea:

Build a neural-network-based Network Intrusion Detection System (NIDS) and introduce a custom structural adaptation mechanism in which low-importance hidden neurons are identified, their useful learned information is transferred into compatible stronger neurons, and the weaker neurons are physically removed from the network.

The project must compare:

1. Standard dense ANN
2. Random neuron removal
3. Conventional structured pruning
4. PREDATOR NET neuron absorption

The objective is NOT simply to maximize classification accuracy.

The actual research objective is:

> Can a neural network reduce its hidden-layer complexity through knowledge-preserving neuron absorption while maintaining intrusion-detection performance?

Primary hypothesis:

> PREDATOR NET should be able to reduce hidden neurons, trainable parameters, and potentially inference cost while keeping detection performance close to the original dense ANN.

Do not assume this hypothesis is true.

The code must experimentally determine whether it is true.

---

# 2. IMPORTANT RESEARCH POSITIONING

Do NOT claim:

- “Nobody has ever done neuron merging.”
- “This is the first neural network that removes neurons.”
- “This is biologically identical to neurons.”
- “The system autonomously understands cybersecurity.”
- “The system protects a real network” unless live prevention is actually implemented.
- “The model is better” before experiments establish this.

Existing literature already includes:

- neural-network pruning,
- structured pruning,
- knowledge transfer,
- neuron merging,
- function-preserving transformations.

In particular, previous work on neuron merging has explicitly studied merging neurons to compensate for information loss caused by pruning, including cosine-similarity-based approaches. Therefore PREDATOR NET should differentiate itself through its complete IDS-oriented adaptive pipeline and its experimentally defined absorption policy, rather than claiming that the broad concept of neuron merging is unprecedented.

PyTorch also provides structured pruning functionality and mechanisms for implementing custom pruning methods, so the implementation should build a genuine custom mechanism rather than pretending that standard pruning APIs alone constitute the research contribution.

Relevant references:

- CIC-IDS2017 official dataset documentation:
  https://www.unb.ca/cic/datasets/ids-2017.html
- Sharafaldin, Lashkari, Ghorbani, 2018:
  “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization”
- Kim et al., 2020:
  “Neuron Merging: Compensating for Pruned Neurons”
- Chen, Goodfellow, Shlens, 2016:
  “Net2Net: Accelerating Learning via Knowledge Transfer”
- He & Xiao, 2024:
  “Structured Pruning for Deep Convolutional Neural Networks: A Survey”

---

# 3. SYSTEM SCOPE

The initial system must perform:

**Network-flow classification → Normal or Attack**

Do NOT begin with packet-level deep learning.

Do NOT begin with live packet interception.

Do NOT begin with multiclass attack classification.

Do NOT begin with a giant neural architecture.

First build a scientifically reproducible binary IDS.

The complete project can later support:

Normal
↓
Attack
↓
Attack family/type

But the first objective is binary intrusion detection.

---

# 4. DATASET

Primary dataset:

**CIC-IDS2017**

Source:

Canadian Institute for Cybersecurity, University of New Brunswick.

The official dataset contains labeled network traffic generated over five working days. The traffic includes benign activity and multiple attacks, and the machine-learning dataset contains flow-level features generated using CICFlowMeter.

The dataset includes traffic associated with protocols such as:

- HTTP
- HTTPS
- FTP
- SSH
- email traffic

and attack scenarios such as:

- FTP brute force
- SSH brute force
- DoS
- DDoS
- Heartbleed
- Web attack
- Infiltration
- Botnet
- Port scanning

according to the official dataset documentation.

IMPORTANT:

A row represents a **network flow**, not an individual raw packet.

Therefore do not describe the model as:

“the neuron receives one packet containing an IP and port.”

Instead describe:

“the model receives numerical and categorical network-flow characteristics derived from captured traffic.”

---

# 5. IP ADDRESS HANDLING

The original CIC-IDS2017 material contains source/destination addressing information, ports, timestamps and protocol information, but the exact columns differ between the labeled-flow and machine-learning CSV exports. The implementation must inspect the actual files and NEVER assume an `IP Address` column exists.

Do NOT make raw source IP and destination IP mandatory model features in the primary experiment.

Reason:

Attackers and victims in the dataset occupy specific hosts, so a model can potentially memorize dataset-specific identities instead of learning general traffic behavior.

Primary model inputs should therefore favor behavioral flow characteristics.

Ports can be used because they are traffic attributes, but they should be evaluated carefully as well.

Create two feature modes:

## MODE A — BEHAVIORAL

Use:

- destination port
- protocol if available
- flow duration
- packet counts
- byte counts
- packet length statistics
- packet rates
- inter-arrival times
- TCP flags
- subflow statistics
- active/idle statistics
- forward/backward traffic statistics

This is the primary research mode.

## MODE B — ENRICHED

Optionally include:

- source port
- destination port
- anonymized source identifier
- anonymized destination identifier
- temporal/network context features

But raw IP strings must NOT simply be converted to arbitrary integer IDs such as:

10.0.0.5 → 1
10.0.0.6 → 2

because that creates meaningless numerical relationships.

If IP experiments are included, use one of:

- hashing,
- categorical embeddings,
- subnet-level features,
- frequency/count encoding,
- behavior-derived IP statistics.

Keep these experiments separate from the main behavioral experiment.

---

# 6. DATA INGESTION PIPELINE

Create a robust loader.

It must:

1. Locate all CIC-IDS2017 CSV files.
2. Read files safely.
3. Strip whitespace from column names.
4. Normalize inconsistent column names.
5. Detect the label column automatically.
6. Report duplicate columns.
7. Detect NaN.
8. Detect positive/negative infinity.
9. Detect duplicated rows.
10. Detect constant columns.
11. Report class distribution.
12. Report feature data types.
13. Save a data-quality report.

The loader must NOT silently alter the dataset.

Every transformation must be logged.

Create:

`artifacts/data_quality_report.json`

containing:

- file names
- rows per file
- columns
- missing values
- infinite values
- duplicate rows
- constant columns
- label distribution
- data types
- preprocessing decisions

---

# 7. DATA SPLITTING

Data leakage is one of the biggest threats to credibility.

Do not randomly split the complete dataset and immediately claim realistic generalization.

Recent 2026 work evaluating CIC-IDS2017 specifically highlights how split methodology and leakage can substantially change reported IDS performance, and recommends leakage-aware evaluation.

Implement at least TWO evaluation protocols.

## PROTOCOL A — STANDARD STRATIFIED SPLIT

For development:

Train:
70–80%

Validation:
10–15%

Test:
10–15%

Use stratification on the binary label.

This is useful for quick experimentation.

## PROTOCOL B — LEAKAGE-AWARE SPLIT

Preferred scientific evaluation.

Where suitable identifiers are available, prevent the same connection/group from being represented across train and test.

Possible grouping hierarchy:

1. duplicate-row grouping
2. flow/connection grouping
3. five-tuple grouping

Five-tuple:

```text
source IP
destination IP
source port
destination port
protocol
```

Do not force this grouping if the necessary fields are not present in the selected CSV; use the strongest defensible available grouping.

## OPTIONAL TEMPORAL PROTOCOL

Because CIC-IDS2017 is naturally organized across Monday-Friday traffic, also implement an optional chronological experiment.

Possible configuration:

```text
Earlier traffic → train
Later traffic → validation/test
```

Do not automatically assume that a day-based split is perfect.

Report exactly what split was used.

---

# 8. PREPROCESSING

Pipeline:

```text
Raw CSV
   ↓
Column normalization
   ↓
Remove invalid records
   ↓
Handle NaN / Infinity
   ↓
Remove constant/duplicate columns
   ↓
Encode categorical features
   ↓
Feature selection
   ↓
Train/Val/Test split
   ↓
Fit scaler ONLY on train
   ↓
Transform validation
   ↓
Transform test
   ↓
Save processed data
```

CRITICAL:

The scaler must NEVER be fitted using test data.

Use:

```python
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
```

Never:

```python
scaler.fit(all_data)
```

---

# 9. NUMERICAL CLEANING

For every numerical feature:

- detect NaN
- detect +Infinity
- detect -Infinity
- detect impossible values where applicable
- convert to float32 after preprocessing

Do not casually remove outliers.

In cybersecurity data, anomalous values can be the actual signal representing attacks.

A preprocessing step that removes every statistical outlier could destroy the information the IDS is supposed to detect.

---

# 10. FEATURE ENGINEERING

Do not create hundreds of artificial features initially.

The first model should use original flow statistics.

Potential feature groups:

## Flow volume

- total bytes
- forward bytes
- backward bytes
- total packets
- forward packets
- backward packets

## Temporal behavior

- flow duration
- forward IAT
- backward IAT
- total IAT
- active time
- idle time

## Packet statistics

- minimum packet length
- maximum packet length
- mean packet length
- standard deviation
- variance
- forward packet length statistics
- backward packet length statistics

## Traffic rate

- bytes/sec
- packets/sec
- forward packets/sec
- backward packets/sec

## TCP behavior

- SYN count
- ACK count
- FIN count
- RST count
- PSH count
- URG count

## Subflow behavior

- subflow packet counts
- subflow byte counts
- forward/backward subflow statistics

## Port/protocol

- destination port
- source port if available
- protocol

The official CIC-IDS2017 page describes the dataset as having more than 80 network-flow features generated by CICFlowMeter.

---

# 11. FEATURE SELECTION

Create a feature-selection module.

Do NOT blindly reduce the dataset to exactly 20 features.

The number of features should be experimentally determined.

Implement:

### Method 1 — Correlation filtering

Remove highly redundant numerical features using a configurable threshold.

Default:

```text
correlation_threshold = 0.95
```

### Method 2 — Mutual information

Calculate:

```text
mutual_information(feature, label)
```

Rank features.

### Method 3 — Optional model-based importance

Use a lightweight reference model such as Random Forest.

The ANN experiment should then test:

```text
all valid features
vs
selected features
```

Save:

```text
artifacts/features/feature_rankings.csv
artifacts/features/selected_features.json
artifacts/features/correlation_matrix.png
artifacts/features/feature_importance.png
```

---

# 12. DATA STORAGE

Because CIC-IDS2017 is large, do not repeatedly keep the complete raw CSV dataset in memory.

Preferred storage:

```text
data/
    raw/
    processed/
        train.parquet
        val.parquet
        test.parquet
```

For high-speed training, additionally create memory-mapped NumPy arrays:

```text
train_X.npy
train_y.npy
val_X.npy
val_y.npy
test_X.npy
test_y.npy
```

Use float32.

Do not convert the complete dataset into Python lists.

---

# 13. BASELINE ANN

Implement a clean PyTorch MLP.

Initial architecture:

```text
Input
  ↓
Linear(input_dim, 128)
  ↓
ReLU
  ↓
Linear(128, 64)
  ↓
ReLU
  ↓
Linear(64, 32)
  ↓
ReLU
  ↓
Linear(32, 1)
```

Call this:

**BaselineMLP**

Do not use BatchNorm initially.

Do not use Dropout initially.

Do not use an excessively deep architecture.

The purpose of the baseline is to provide a stable controlled reference for PREDATOR.

---

# 14. OUTPUT

Use:

```python
output = Linear(32, 1)
```

The output is a logit.

Use:

```python
BCEWithLogitsLoss
```

instead of applying sigmoid inside the model.

During inference:

```python
probability = torch.sigmoid(logit)
```

Classification:

```python
probability >= threshold
→ ATTACK

probability < threshold
→ BENIGN
```

The default threshold can initially be:

```text
0.5
```

but threshold tuning must be performed on validation data.

Do not tune the threshold using test data.

---

# 15. CLASS IMBALANCE

CIC-IDS2017 is not naturally perfectly balanced.

Therefore calculate:

```text
negative_count
positive_count
```

from training data.

For binary classification support:

```text
pos_weight = negative_count / positive_count
```

and optionally use:

```python
BCEWithLogitsLoss(pos_weight=...)
```

However, do not assume the resulting weighting is optimal.

Implement configuration:

```yaml
class_weighting:
    enabled: true
    method: pos_weight
```

Add an optional unweighted experiment.

Compare both.

---

# 16. BASELINE TRAINING

Use:

```text
Optimizer:
AdamW

Initial learning rate:
1e-3

Weight decay:
1e-4

Activation:
ReLU

Epochs:
configurable, e.g. 30–100

Early stopping:
enabled

Patience:
configurable

Seed:
fixed
```

Use deterministic seeds where practical.

Save:

```text
artifacts/models/baseline_best.pt
```

Also save:

```text
baseline_metrics.json
baseline_confusion_matrix.png
baseline_training_curve.png
baseline_roc_curve.png
baseline_pr_curve.png
```

---

# 17. BASELINE METRICS

Never report only accuracy.

Report:

- Accuracy
- Precision
- Recall
- F1
- Macro-F1
- Balanced Accuracy
- ROC-AUC
- PR-AUC
- False Positive Rate
- False Negative Rate
- Confusion Matrix
- Inference latency
- Throughput
- Number of parameters
- Model size

Cybersecurity emphasis:

The most important metrics should include:

```text
Recall
Precision
F1
False Positive Rate
PR-AUC
```

because an IDS can be misleading if it achieves high accuracy simply by classifying most traffic as benign.

---

# 18. CORE PREDATOR CONCEPT

The neural network consists of:

```text
Input Features
       ↓
Hidden Layer 1
       ↓
Hidden Layer 2
       ↓
Hidden Layer 3
       ↓
Output
```

PREDATOR operates ONLY on hidden neurons.

Do not remove:

- input features during neuron absorption
- output neurons

The initial structure could be:

```text
67-ish features
      ↓
128 neurons
      ↓
64 neurons
      ↓
32 neurons
      ↓
1 output
```

The exact input dimension must be determined dynamically after preprocessing.

---

# 19. WHAT IS A NEURON?

For a fully connected layer:

```text
h = ReLU(Wx + b)
```

Each hidden neuron has:

- incoming weights
- bias
- outgoing weights
- activation behavior

For a hidden layer with `N` neurons:

```text
Neuron 0
Neuron 1
Neuron 2
...
Neuron N-1
```

PREDATOR calculates an importance score for each hidden neuron.

---

# 20. NEURON IMPORTANCE

Do NOT define neuron strength arbitrarily.

Implement a measurable importance system.

Use a first-order contribution criterion.

For a hidden neuron's activation:

```text
a_j
```

and its gradient:

```text
∂L / ∂a_j
```

define:

```text
importance_j =
mean(
    |a_j * ∂L/∂a_j|
)
```

over a calibration subset of training/validation batches.

Interpretation:

A neuron that has little activation or little influence on the loss tends to receive a lower importance score.

This becomes the main behavioral importance metric.

---

# 21. SECONDARY IMPORTANCE SIGNAL

Also calculate outgoing weight magnitude.

For neuron j:

```text
weight_importance_j =
||outgoing_weights_j||_1
```

Normalize it.

Then optionally combine:

```text
combined_score =
α * normalized_gradient_importance
+
(1-α) * normalized_weight_importance
```

Default:

```text
α = 0.7
```

Keep all coefficients configurable.

---

# 22. IMPORTANCE CALIBRATION

Do not calculate neuron importance using the entire training set every time.

Use a calibration subset:

```text
calibration_size:
configurable
e.g. 10,000–100,000 samples
```

The exact number should depend on available hardware.

Save the importance results:

```text
artifacts/predator/neuron_importance.csv
```

with columns:

```text
layer
neuron_index
activation_importance
weight_importance
combined_importance
rank
status
```

Example:

```text
layer2,17,0.0031,0.021,0.0084,61,weak
layer2,42,0.2114,0.682,0.352,2,strong
```

---

# 23. STRONG VS WEAK

Do not hard-code:

```text
importance < 0.5 = weak
```

because importance scales are model-dependent.

Instead support configurable policies:

### Percentile policy

Example:

```text
bottom 15% = weak candidates
top 20% = strong candidates
```

### Relative policy

Example:

```text
importance_j < median * weak_factor
```

### Hybrid policy

Use percentile + minimum activity threshold.

Default implementation:

```text
weak_candidate_percentile = 15%
strong_candidate_percentile = 30%
```

These are candidates, not automatically removable neurons.

---

# 24. WHY THE STRONG NEURON NEEDS A COMPATIBLE WEAK NEURON

The weak neuron should NOT randomly transfer information into any strong neuron.

Find a compatible strong neuron.

Define:

```text
activation_profile_j
```

as the activation vector of neuron j across calibration samples.

For weak neuron `w` and strong candidate `s`:

```text
activation_similarity =
cosine_similarity(H_w, H_s)
```

Also calculate:

```text
incoming_weight_similarity =
cosine_similarity(W_w, W_s)
```

Then:

```text
compatibility =
β * activation_similarity
+
(1-β) * incoming_weight_similarity
```

Default:

```text
β = 0.7
```

Select the strong neuron with maximum compatibility among the allowed strong neurons.

---

# 25. IMPORTANT RESTRICTION

Only absorb a weak neuron if:

```text
compatibility >= minimum_similarity
```

Example configuration:

```yaml
minimum_similarity: 0.70
```

But this must be experimentally configurable.

If no strong compatible neuron exists:

```text
DO NOT ABSORB.
```

Instead record:

```text
absorption_status = rejected_no_compatible_neuron
```

This prevents random destructive compression.

---

# 26. ABSORPTION MECHANISM — MVP VERSION

Start with the conservative mechanism.

Suppose:

```text
weak neuron = w
strong neuron = s
```

Their outgoing weight vectors are:

```text
U_w
U_s
```

Calculate:

```text
α =
clip(
    compatibility *
    importance_w /
    (importance_s + epsilon),
    alpha_min,
    alpha_max
)
```

Example defaults:

```text
alpha_min = 0.05
alpha_max = 0.50
```

Then:

```text
U_s_new = U_s + α * U_w
```

The weak neuron's outgoing information is therefore partially transferred into the stronger neuron.

Then remove neuron `w`.

Do NOT immediately assume that this operation preserves accuracy.

Validate it.

---

# 27. OPTIONAL FULL ABSORPTION

Create an experimental second strategy.

Instead of only changing outgoing weights, also modify the incoming weights.

For:

```text
V_w
V_s
```

use:

```text
V_s_new =
(1-γ) * V_s
+
γ * V_w
```

and:

```text
b_s_new =
(1-γ) * b_s
+
γ * b_w
```

where:

```text
γ = configurable absorption coefficient
```

This creates:

**PREDATOR-FULL**

while the conservative version is:

**PREDATOR-LIGHT**

Compare them experimentally.

Do not assume PREDATOR-FULL will perform better.

---

# 28. STRUCTURAL REMOVAL

This is extremely important.

Do not simply set neuron weights to zero and claim the network became smaller.

That would be masking/pruning, not actual structural removal.

After absorption, physically rebuild the adjacent Linear layers.

Suppose:

```text
Layer A:
input_dim → 128

Layer B:
128 → 64
```

Removing neuron 17 from Layer A's output means:

### Layer A

Remove row 17 from:

```text
LayerA.weight
```

Remove:

```text
LayerA.bias[17]
```

So:

```text
128 → 127
```

### Layer B

Because its input dimension was 128, remove column 17:

```text
LayerB.weight[:,17]
```

Therefore:

```text
127 → 64
```

The model is now physically smaller.

This must happen recursively for whichever hidden layer is modified.

---

# 29. DYNAMIC MLP ARCHITECTURE

Implement:

```python
DynamicMLP
```

with:

```python
nn.ModuleList
```

instead of hard-coding individual layer dimensions everywhere.

Example:

```text
input_dim
   ↓
Linear(input_dim, 128)
   ↓
ReLU
   ↓
Linear(128, 64)
   ↓
ReLU
   ↓
Linear(64, 32)
   ↓
ReLU
   ↓
Linear(32, 1)
```

Architecture configuration:

```yaml
hidden_dims:
  - 128
  - 64
  - 32
```

PREDATOR should be able to turn this into:

```yaml
hidden_dims:
  - 113
  - 58
  - 29
```

without rebuilding the entire codebase.

---

# 30. ABSORPTION ACCEPTANCE RULE

This is one of the most important parts of the research design.

Never remove neurons solely because their importance is low.

For each candidate:

```text
1. Identify weak neuron
2. Identify compatible strong neuron
3. Create temporary absorbed network
4. Measure validation performance
5. Compare against current model
6. Accept or rollback
```

Example:

```text
Current validation F1:
0.941

After absorption:
0.939

Allowed degradation:
0.005

Accept:
YES
```

But:

```text
Current:
0.941

After:
0.920

Drop:
0.021

Accept:
NO
```

The candidate is rolled back.

---

# 31. PERFORMANCE TOLERANCE

Make it configurable:

```yaml
max_validation_f1_drop: 0.005
```

Also optionally constrain:

```yaml
max_validation_recall_drop: 0.005
```

Therefore a neuron cannot be removed if it causes unacceptable degradation.

This makes PREDATOR a **validation-gated adaptive compression system** rather than a blind pruning system.

---

# 32. KNOWLEDGE DISTILLATION

Add a stronger optional version.

Before compression:

```text
Teacher = original baseline ANN
```

After neuron absorption:

```text
Student = smaller PREDATOR ANN
```

Train the smaller network using:

```text
hard labels
+
teacher predictions
```

Loss:

```text
L_total =
(1-λ) * L_classification
+
λ * L_distillation
```

For binary classification, implement a stable distillation loss using teacher/student logits.

For multiclass mode later, use temperature-scaled softmax/KL divergence.

Default:

```text
temperature = 2.0
lambda = 0.3
```

Make these configurable.

Name this experiment:

**PREDATOR-D**

This provides a stronger interpretation of “knowledge preservation.”

---

# 33. ITERATIVE PREDATOR LOOP

Implement:

```text
Train baseline
      ↓
Evaluate
      ↓
Calculate neuron importance
      ↓
Identify weak candidates
      ↓
Select weak neuron
      ↓
Find compatible strong neuron
      ↓
Simulate absorption
      ↓
Physically remove neuron
      ↓
Validate
      ↓
 ┌───────────────┐
 │ Performance OK│
 └───────┬───────┘
         │
       YES
         ↓
     Accept
         ↓
     Fine-tune
         ↓
Calculate importance again
         ↓
Next neuron
```

If validation performance becomes unacceptable:

```text
Rollback latest change
```

Stop when:

```text
minimum network width reached
OR
no valid absorption exists
OR
performance tolerance exceeded
OR
maximum absorption iterations reached
```

---

# 34. MINIMUM NETWORK WIDTH

Never allow:

```text
128 → 0
```

or:

```text
1 hidden neuron
```

unless specifically testing extreme compression.

Default:

```yaml
minimum_hidden_width:
  layer_1: 16
  layer_2: 8
  layer_3: 8
```

These values are experimental defaults.

The agent should never claim they are theoretically optimal.

---

# 35. PREDATOR ALGORITHM

Implement a clean function conceptually equivalent to:

```python
predator_step(
    model,
    calibration_loader,
    validation_loader,
    target_layer,
    config
)
```

It should return:

```text
new_model
absorption_record
accepted/rejected
validation_metrics_before
validation_metrics_after
```

---

# 36. PREDATOR PSEUDOCODE

```text
INPUT:
    trained baseline model M
    calibration dataset C
    validation dataset V

REPEAT:

    1. Collect hidden-layer activations on C

    2. Calculate importance score for every hidden neuron

    3. Select low-importance neuron w

    4. Select strong-neuron candidates

    5. Calculate compatibility between w and each strong candidate

    6. Choose compatible strong neuron s

    7. If compatibility is below threshold:
           reject candidate
           continue

    8. Create a candidate model M'

    9. Transfer part of w's outgoing information into s

   10. Remove w structurally from the network

   11. Evaluate M' on validation data

   12. If performance degradation <= tolerance:
           accept M'
           optionally fine-tune
       else:
           rollback

   13. Recalculate neuron importance

UNTIL stopping condition

RETURN final PREDATOR model
```

---

# 37. LAYER SELECTION STRATEGY

Initially perform absorption independently in each hidden layer.

Possible order:

```text
Layer 3
→ Layer 2
→ Layer 1
```

Why?

Earlier layers provide more general representations, while deeper layers are closer to the classification decision.

But do NOT assume this is universally optimal.

Implement configuration:

```yaml
layer_strategy:
  mode: deepest_first
```

Also support:

```text
sequential
weakest_first
round_robin
```

This allows an ablation experiment.

---

# 38. ABSORPTION RATE

Do not remove 50% of the network instantly.

Use small iterative changes.

Example:

```text
Remove 1 neuron
→ fine-tune
→ evaluate
→ remove another
→ fine-tune
→ evaluate
```

This is slower but scientifically much easier to analyze.

Create a faster experimental mode later:

```text
batch_absorption
```

where several candidates are processed together.

---

# 39. FINE-TUNING AFTER ABSORPTION

After each accepted absorption:

```text
fine-tune for 1–5 epochs
```

or use:

```yaml
fine_tune:
    enabled: true
    epochs: 3
    learning_rate: 0.0001
```

Do not restart the model from scratch.

Use the absorbed model's current parameters.

---

# 40. BASELINE VS PREDATOR EXPERIMENT

This is the centerpiece.

## MODEL A

```text
Dense ANN
```

Example:

```text
128 → 64 → 32
```

## MODEL B

```text
Random structured removal
```

## MODEL C

```text
Magnitude-based structured pruning
```

PyTorch provides structured pruning utilities based on norms and also supports custom pruning methods.

## MODEL D

```text
PREDATOR-LIGHT
```

## MODEL E

```text
PREDATOR-FULL
```

## MODEL F

```text
PREDATOR-D
```

PREDATOR-D = absorption + knowledge distillation.

---

# 41. CORE RESEARCH TABLE

Produce:

| Model | Neurons | Parameters | Accuracy | Precision | Recall | F1 | PR-AUC | FPR | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense ANN | | | | | | | | | |
| Random pruning | | | | | | | | | |
| Structured pruning | | | | | | | | | |
| PREDATOR-Light | | | | | | | | | |
| PREDATOR-Full | | | | | | | | | |
| PREDATOR-D | | | | | | | | | |

Do not populate any cell until actual experiments produce the value.

---

# 42. COMPRESSION METRICS

Define:

```text
neuron_reduction =
1 - final_hidden_neurons /
    baseline_hidden_neurons
```

Define:

```text
parameter_reduction =
1 - final_parameters /
    baseline_parameters
```

Define:

```text
model_size_reduction =
1 - final_model_size /
    baseline_model_size
```

Potential efficiency metric:

```text
efficiency_score =
F1 / parameter_count
```

But do not use this as the primary research metric.

Keep raw metrics visible.

---

# 43. INFERENCE EVALUATION

Measure actual inference.

Use identical:

- hardware
- batch size
- data preprocessing
- PyTorch version
- number of runs

Measure:

```text
single-sample latency
batch latency
samples/sec
```

Warm up the model before timing.

Use multiple repetitions.

Report:

```text
mean
median
standard deviation
p95 latency
```

Do not claim that parameter reduction automatically produced speedup.

Actual acceleration depends on implementation and hardware.

Structured pruning is generally more suitable for practical hardware acceleration than unstructured sparsity because it produces compact tensor structures rather than irregular sparse weights.

---

# 44. IMPORTANT DISTINCTION

PREDATOR has two separate achievements:

## Research property

```text
Can the network preserve useful predictive behavior
while losing neurons?
```

## Engineering property

```text
Does the resulting smaller dense architecture
actually require less computation?
```

The first can succeed while the second shows little improvement.

Both must therefore be measured separately.

---

# 45. ABLATION STUDIES

Implement at least these.

## Ablation 1 — No absorption knowledge transfer

Remove weak neurons only.

Compare against:

```text
PREDATOR absorption
```

Purpose:

Determine whether information transfer matters.

---

## Ablation 2 — Random strong-neuron target

Instead of compatibility:

```text
random strong neuron
```

Purpose:

Determine whether compatibility selection matters.

---

## Ablation 3 — Weight similarity only

Use:

```text
incoming-weight cosine similarity
```

without activation similarity.

---

## Ablation 4 — Activation similarity only

Use:

```text
activation cosine similarity
```

without weight similarity.

---

## Ablation 5 — No distillation

Compare:

```text
PREDATOR
```

vs

```text
PREDATOR + distillation
```

---

## Ablation 6 — Different importance scores

Compare:

```text
weight magnitude
vs
activation-gradient contribution
vs
combined score
```

---

# 46. ROBUSTNESS EXPERIMENT

The model should not only be tested on clean data.

Create optional perturbation experiments.

Examples:

- modest feature noise
- packet-count perturbation
- timing perturbation
- byte-count perturbation
- missing-feature simulation

Purpose:

Test whether the compressed network becomes disproportionately fragile.

Do NOT create unrealistic random modifications that have no cybersecurity interpretation.

---

# 47. FEATURE ABLATION

Remove selected groups:

```text
without ports
without TCP flags
without timing
without packet statistics
without byte statistics
```

Measure performance.

This helps answer:

> What traffic information does the ANN actually use?

It also connects the feature-level behavior with the neuron-level behavior.

---

# 48. NEURON EVOLUTION LOG

Every accepted or rejected event must be recorded.

Create:

```text
artifacts/predator/absorption_history.csv
```

Columns:

```text
iteration
layer
weak_neuron
strong_neuron
weak_importance
strong_importance
activation_similarity
weight_similarity
compatibility
alpha
f1_before
f1_after
recall_before
recall_after
parameter_count_before
parameter_count_after
decision
reason
```

Example:

```text
17,layer2,22,11,0.004,0.31,0.87,0.78,0.84,0.11,
0.941,0.939,0.955,0.954,16256,15800,accepted,
within_tolerance
```

These numbers must come from actual execution.

---

# 49. VISUALIZATION

Create a network-evolution visualization.

Before:

```text
● ● ● ● ● ● ● ●
● ● ● ● ● ●
● ● ● ●
```

During:

```text
Weak ● ─────→ Strong ●
        absorb
```

After:

```text
● ● ● ● ●
● ● ● ●
● ● ●
```

Generate:

### Graph 1

Neurons vs iteration

### Graph 2

Parameters vs iteration

### Graph 3

Validation F1 vs neuron reduction

### Graph 4

Recall vs neuron reduction

### Graph 5

Inference latency vs neuron reduction

### Graph 6

Neuron importance distribution

### Graph 7

Accepted vs rejected absorption attempts

### Graph 8

Neuron survival timeline

---

# 50. NEURON SURVIVAL VISUALIZATION

Assign each neuron a lifecycle:

```text
ACTIVE
  ↓
WEAK CANDIDATE
  ↓
ABSORBED
  ↓
REMOVED
```

For strong neurons:

```text
ACTIVE
  ↓
ABSORBER
  ↓
ABSORBER + UPDATED
```

This gives the project its distinctive PREDATOR identity.

---

# 51. OPTIONAL SOC DEMO

Do NOT initially build real packet blocking.

Create a simulated live NIDS.

Input:

```text
CIC-IDS2017 test flows
```

Stream rows one-by-one.

For each flow display:

```text
Flow ID
Protocol
Source Port
Destination Port
Flow Duration
Packet Count
Byte Count
Prediction
Confidence
```

Example:

```text
────────────────────────────────
PREDATOR NET — LIVE FLOW MONITOR
────────────────────────────────

Protocol      TCP
Dst Port     443
Duration     0.018 sec
Packets      14
Bytes        4210

Prediction:
⚠ ATTACK

Confidence:
94.2%

Active neurons:
96

Absorbed:
32
────────────────────────────────
```

This is a replayed dataset demo, not proof of live network protection.

---

# 52. OPTIONAL LIVE MODE

Only after the complete offline experiment succeeds.

Possible architecture:

```text
Live traffic
     ↓
Flow collector
     ↓
Flow feature extraction
     ↓
Feature normalization
     ↓
PREDATOR NET
     ↓
Normal / Attack
     ↓
Alert
```

Possible future components:

- Zeek
- CICFlowMeter
- packet capture
- flow collector
- FastAPI
- WebSocket
- Streamlit

But keep this out of the core research experiment.

---

# 53. TECHNOLOGY STACK

Use:

## Language

Python 3.11+

## Deep learning

PyTorch

## Data processing

Polars or pandas

## Numerical computing

NumPy

## Machine learning utilities

scikit-learn

## Visualization

Matplotlib

Optional:

- Seaborn
- Plotly
- Streamlit
- FastAPI
- Uvicorn

Do not introduce unnecessary frameworks.

---

# 54. RECOMMENDED PROJECT STRUCTURE

```text
predator-net/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── config/
│   ├── baseline.yaml
│   ├── predator.yaml
│   └── experiments.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── src/
│   └── predator_net/
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── loader.py
│       │   ├── cleaner.py
│       │   ├── splitter.py
│       │   ├── scaler.py
│       │   └── dataset.py
│       │
│       ├── features/
│       │   ├── selection.py
│       │   ├── engineering.py
│       │   └── analysis.py
│       │
│       ├── models/
│       │   ├── baseline_mlp.py
│       │   ├── dynamic_mlp.py
│       │   └── model_utils.py
│       │
│       ├── predator/
│       │   ├── importance.py
│       │   ├── similarity.py
│       │   ├── absorption.py
│       │   ├── structural_rebuild.py
│       │   ├── distillation.py
│       │   └── controller.py
│       │
│       ├── training/
│       │   ├── trainer.py
│       │   ├── losses.py
│       │   ├── callbacks.py
│       │   └── seed.py
│       │
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── benchmark.py
│       │   ├── latency.py
│       │   └── ablation.py
│       │
│       ├── visualization/
│       │   ├── plots.py
│       │   ├── network_graph.py
│       │   └── evolution.py
│       │
│       └── demo/
│           ├── replay.py
│           └── dashboard.py
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── train_baseline.py
│   ├── run_predator.py
│   ├── run_benchmarks.py
│   └── generate_report.py
│
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_importance.py
│   ├── test_similarity.py
│   ├── test_absorption.py
│   └── test_structural_rebuild.py
│
└── artifacts/
    ├── data_quality/
    ├── features/
    ├── models/
    ├── predator/
    ├── evaluation/
    └── figures/
```

---

# 55. CONFIGURATION FILE

Create:

```yaml
project:
  name: PREDATOR_NET
  seed: 42

data:
  dataset: CIC-IDS2017
  label_column: auto

features:
  mode: behavioral
  correlation_threshold: 0.95
  mutual_information: true

model:
  hidden_dims:
    - 128
    - 64
    - 32
  activation: relu

training:
  batch_size: 1024
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 50
  early_stopping_patience: 8

predator:
  enabled: true
  weak_percentile: 15
  strong_percentile: 30
  minimum_similarity: 0.70

  importance:
    gradient_weight: 0.70
    weight_weight: 0.30

  absorption:
    alpha_min: 0.05
    alpha_max: 0.50

  validation:
    max_f1_drop: 0.005
    max_recall_drop: 0.005

  minimum_width:
    layer_1: 16
    layer_2: 8
    layer_3: 8

distillation:
  enabled: false
  temperature: 2.0
  lambda: 0.3
```

Every important experimental parameter must be configurable.

---

# 56. COMMAND-LINE INTERFACE

Implement:

```bash
python scripts/prepare_dataset.py
```

Then:

```bash
python scripts/train_baseline.py
```

Then:

```bash
python scripts/run_predator.py
```

Then:

```bash
python scripts/run_benchmarks.py
```

Then:

```bash
python scripts/generate_report.py
```

Optional:

```bash
streamlit run src/predator_net/demo/dashboard.py
```

---

# 57. REPRODUCIBILITY

Every experiment should save:

```text
config
seed
dataset version/details
feature list
split information
model architecture
training parameters
metrics
runtime
hardware
software versions
```

Save:

```text
experiment_metadata.json
```

Example:

```json
{
  "seed": 42,
  "torch_version": "...",
  "python_version": "...",
  "device": "cuda",
  "feature_count": 67,
  "hidden_dims": [128,64,32]
}
```

---

# 58. HARDWARE SUPPORT

Automatically detect:

```python
cuda
```

if available.

Fallback:

```text
CPU
```

Use mixed precision on CUDA where appropriate.

Do not make the project dependent on a specific GPU.

---

# 59. UNIT TESTS

Write tests before declaring the mechanism complete.

### Test 1

Removing neuron correctly changes layer dimensions.

### Test 2

Forward pass still works after neuron removal.

### Test 3

Parameter count decreases after removal.

### Test 4

Output shape remains:

```text
(batch, 1)
```

### Test 5

Absorption correctly modifies strong-neuron outgoing weights.

### Test 6

Rollback restores the exact previous model.

### Test 7

No neuron below minimum width can be removed.

### Test 8

No input neuron is accidentally removed.

### Test 9

No output neuron is accidentally removed.

### Test 10

Saving and loading an absorbed model produces the same predictions.

---

# 60. MODEL INTEGRITY TEST

Before and after each structural change:

```text
model.forward(X)
```

must succeed.

Check:

```text
NaN predictions
Inf predictions
incorrect dimensions
exploding logits
```

Abort an absorption step if numerical instability occurs.

---

# 61. EXPERIMENTAL PHASES

The coding agent must implement the project in phases.

## PHASE 1

Dataset discovery.

Deliver:

```text
data_quality_report
```

Nothing else.

---

## PHASE 2

Preprocessing.

Deliver:

```text
train
validation
test
feature metadata
```

---

## PHASE 3

Baseline ANN.

Deliver:

```text
trained baseline
metrics
plots
```

Do not implement PREDATOR until baseline is working.

---

## PHASE 4

Neuron instrumentation.

Implement:

- activation capture
- gradient capture
- neuron importance

Visualize:

```text
neuron importance distribution
```

---

## PHASE 5

Compatibility system.

Implement:

- activation vectors
- cosine similarity
- incoming-weight similarity
- compatibility score

Test this independently.

---

## PHASE 6

Structural absorption.

Implement:

```text
weak → strong → transfer → remove → rebuild
```

Test using a tiny artificial network before applying it to CIC-IDS2017.

---

## PHASE 7

Validation gate.

Implement:

```text
candidate → evaluate → accept/reject
```

with rollback.

---

## PHASE 8

Iterative PREDATOR.

Implement repeated absorption + fine-tuning.

---

## PHASE 9

Benchmarking.

Compare:

```text
Baseline
Random pruning
Structured pruning
PREDATOR-Light
PREDATOR-Full
PREDATOR-D
```

---

## PHASE 10

Research analysis.

Generate:

- compression plots
- neuron evolution
- performance curves
- latency comparison
- ablations

---

## PHASE 11

Demo.

Build dataset replay dashboard.

---

## PHASE 12

Research paper materials.

Generate:

- methodology diagram
- algorithm pseudocode
- experiment tables
- graphs
- limitations
- conclusion based only on measured results

---

# 62. EXPECTED FINAL ARCHITECTURAL FLOW

The final system should conceptually look like:

```text
               CIC-IDS2017
                     │
                     ▼
             Data Processing
                     │
                     ▼
            Feature Selection
                     │
                     ▼
             Normalization
                     │
                     ▼
              BASE ANN
                     │
                     ▼
          Network-flow prediction
                     │
        ┌────────────┴────────────┐
        │                         │
      Train                    Measure
        │                       neurons
        │                         │
        └───────────┬─────────────┘
                    ▼
             Importance Score
                    │
            ┌───────┴───────┐
            │               │
         Weak             Strong
        neurons            neurons
            │               │
            └──────┬────────┘
                   ▼
            Compatibility
                   │
                   ▼
       Knowledge-Preserving
             Absorption
                   │
                   ▼
          Structural Removal
                   │
                   ▼
             Smaller ANN
                   │
                   ▼
              Fine-tuning
                   │
                   ▼
              Validation
             /          \
          Accept       Reject
            │             │
            ▼             ▼
       Continue         Rollback
            │
            ▼
      Final PREDATOR
            │
            ▼
       Normal / Attack
```

---

# 63. RESEARCH QUESTION

The paper should revolve around:

> **Can knowledge-preserving neuron absorption reduce the structural complexity of an artificial neural network for network intrusion detection while maintaining comparable predictive performance?**

Secondary questions:

1. Which neuron-importance criterion identifies removable neurons most reliably?
2. Does transferring information to a compatible neuron preserve performance better than direct deletion?
3. Does activation similarity provide a useful criterion for selecting the absorber neuron?
4. How much neuron/parameter reduction can be achieved before performance degradation becomes significant?
5. Does the resulting structural compression produce actual inference benefits?
6. Does knowledge distillation further improve compressed-model performance?

---

# 64. RESEARCH CONTRIBUTIONS

Potential contribution claims should be phrased carefully and only retained if the experiments support them.

Possible contributions:

### Contribution 1

An IDS-oriented adaptive neuron-absorption framework.

### Contribution 2

A compatibility-based absorber-selection mechanism using neuron behavior and learned representations.

### Contribution 3

Validation-gated structural neuron removal.

### Contribution 4

Experimental comparison against conventional neuron removal/pruning.

### Contribution 5

Analysis of the tradeoff:

```text
Detection performance
        vs
Network complexity
```

### Contribution 6

Visualization of neuron survival and absorption dynamics.

Do not state these as “novel” automatically.

Perform a literature review before making novelty claims.

---

# 65. LITERATURE CONTEXT

The literature review should cover four major areas.

## A. Machine Learning for IDS

Neural networks, feed-forward networks, autoencoders, recurrent models, etc.

A comparative study of deep learning approaches for IDS has specifically evaluated feed-forward networks, autoencoders, DBNs and LSTMs on CIC-IDS2017 and related datasets, making a feed-forward MLP a defensible baseline for this project.

---

## B. Neural Network Pruning

Discuss:

- unstructured pruning
- structured pruning
- magnitude pruning
- channel/neuron pruning
- dynamic pruning

Structured pruning removes whole computational structures and is more directly connected to physical architecture reduction and potential hardware acceleration.

---

## C. Neuron Merging

Discuss the work:

**“Neuron Merging: Compensating for Pruned Neurons”**

That work explicitly investigates merging information from neurons to compensate for pruning-related information loss and uses cosine similarity among neurons as part of its methodology.

PREDATOR should acknowledge this literature instead of presenting neuron absorption as an entirely unexplored concept.

---

## D. Knowledge Transfer

Discuss:

**Net2Net**

which introduced function-preserving transformations for transferring information between neural-network architectures.

Then explain how PREDATOR explores the reverse direction:

```text
Net2Net:
Knowledge → larger architecture

PREDATOR:
Knowledge → compact architecture
```

This is a useful conceptual framing, not a claim that PREDATOR is a direct extension of Net2Net.

---

# 66. IMPORTANT DATASET LIMITATION

CIC-IDS2017 is useful but does not equal modern internet traffic.

The project must explicitly state:

```text
The dataset represents a controlled research environment.
```

It should not claim:

```text
This model detects every real-world cyberattack.
```

The official dataset documentation itself describes a controlled network environment with specific victim/attacker infrastructure.

---

# 67. IMPORTANT EVALUATION LIMITATION

Recent 2026 research using CIC-IDS2017 has demonstrated that apparently excellent IDS results can change substantially under leakage-aware evaluation protocols.

Therefore:

Do not optimize the model merely to obtain:

```text
99.9% accuracy
```

Instead emphasize:

```text
generalization
F1
recall
false-positive rate
PR-AUC
compression
latency
```

and clearly describe the split protocol.

---

# 68. RESEARCH PAPER STRUCTURE

Generate the final paper using:

## Abstract

Problem → method → experiment → findings.

Do not write findings before experiments exist.

## 1. Introduction

Explain:

- network intrusion detection
- neural-network complexity
- pruning
- knowledge preservation
- PREDATOR idea

## 2. Related Work

Cover:

- IDS machine learning
- ANN-based IDS
- structured pruning
- neuron merging
- knowledge transfer

## 3. Dataset

Explain:

- CIC-IDS2017
- traffic generation
- flow representation
- features
- attack categories
- limitations

## 4. Methodology

Explain:

- baseline ANN
- neuron importance
- weak/strong identification
- compatibility
- absorption
- structural removal
- fine-tuning
- validation gate
- optional distillation

## 5. Experimental Setup

Describe:

- hardware
- Python/PyTorch
- preprocessing
- splits
- hyperparameters
- metrics

## 6. Results

Only measured numbers.

## 7. Ablation Study

Explain which components actually matter.

## 8. Discussion

Explain:

- why certain neurons were removed
- whether knowledge transfer worked
- compression/performance tradeoff
- failure cases

## 9. Limitations

Include:

- dataset limitations
- possible distribution shift
- IP/address leakage concerns
- computational cost of repeated validation
- heuristic absorption coefficients
- lack of live-network validation

## 10. Conclusion

Answer the research question using measured evidence.

---

# 69. WHAT THE FINAL DEMO SHOULD SHOW

The strongest demonstration should be:

```text
                     PREDATOR NET

Original network
128 → 64 → 32

        ↓
   neuron analysis

Weak neurons identified

        ↓
   knowledge absorption

        ↓

Compressed network
104 → 53 → 29

        ↓

Compare:

Accuracy
Recall
F1
PR-AUC
Parameters
Inference latency

        ↓

NORMAL / ATTACK
```

The visual story should be:

```text
Large model
     ↓
Learns
     ↓
Analyzes neuron usefulness
     ↓
Absorbs useful information
     ↓
Removes unnecessary structure
     ↓
Becomes compact
     ↓
Still detects attacks
```

---

# 70. CRITICAL CODING RULES

The coding agent must follow all of these:

1. Never fabricate metrics.
2. Never fabricate dataset statistics if they can be computed.
3. Never hard-code the input feature count.
4. Never assume raw IP columns exist.
5. Never fit preprocessing on test data.
6. Never modify the test set during model development.
7. Never remove a neuron without structural dimension updates.
8. Never claim parameter reduction if the model is only masked.
9. Never call random deletion “knowledge preservation.”
10. Never absorb neurons without validation.
11. Always support rollback.
12. Preserve random seeds.
13. Save experiment configurations.
14. Log every structural modification.
15. Make every threshold configurable.
16. Start with binary classification.
17. Make multiclass classification an extension.
18. Keep live packet interception outside the core experiment.
19. Test the structural modification on a tiny toy network before CIC-IDS2017.
20. Do not declare the research hypothesis successful until multiple experiments support it.

---

# 71. FIRST IMPLEMENTATION TARGET

Do NOT begin by creating:

- dashboard
- live scanner
- fancy animations
- attack blocking
- complex transformers
- LSTM
- CNN
- reinforcement learning

The first working version MUST be:

```text
CIC-IDS2017
     ↓
Clean
     ↓
Preprocess
     ↓
Train baseline MLP
     ↓
Normal / Attack
     ↓
Metrics
```

After that:

```text
Baseline
     ↓
Neuron importance
     ↓
Weak/Strong identification
     ↓
Compatibility
     ↓
Absorption
     ↓
Structural removal
     ↓
Validation
```

That is the actual heart of the project.

---

# 72. FINAL SUCCESS CONDITION

Do not define project success as:

> “The model has high accuracy.”

Define success as:

> “PREDATOR NET demonstrates measurable structural reduction in a neural intrusion-detection model while preserving acceptable detection performance relative to the dense baseline, and the experiments clearly quantify the tradeoff between compression and detection quality.”

Possible outcome:

```text
Baseline:
Excellent accuracy

PREDATOR:
Much smaller
Almost identical F1

→ Strong result
```

Another possible outcome:

```text
Baseline:
Excellent

PREDATOR:
Smaller
Small F1 degradation

→ Still scientifically useful
```

Another possible outcome:

```text
Baseline:
Excellent

PREDATOR:
Smaller
Large performance degradation

→ Research mechanism needs improvement
```

Even the third outcome is a legitimate experiment.

The system should report the truth rather than optimize for a predetermined conclusion.

---

# 73. CODING AGENT INSTRUCTION

Build this project incrementally.

At the end of every phase:

1. Run tests.
2. Generate artifacts.
3. Display the actual result.
4. Explain failures.
5. Do not silently skip errors.
6. Do not invent placeholder metrics.
7. Keep research code modular.
8. Maintain reproducibility.
9. Do not proceed to the next major phase if the previous phase is fundamentally broken.

Begin with:

```text
PHASE 1:
Dataset inspection + reproducible preprocessing pipeline.
```

Do not implement the PREDATOR mechanism until the baseline ANN successfully trains and produces a valid evaluation report.

The final codebase must be suitable for:

- university project demonstration
- GitHub publication
- research-paper experimentation
- reproducible benchmarking
- later dashboard integration

---

### Research basis

The core technical choices above are anchored in the current literature: CIC-IDS2017's official documentation confirms the flow-oriented dataset and its attack scenarios/features; PyTorch provides structured pruning and custom pruning support; prior neuron-merging work demonstrates that information transfer can compensate for pruning; and recent 2026 CIC-IDS2017 work shows why leakage-aware evaluation matters.

The most important conceptual upgrade is **validation-gated absorption + physical structural rebuilding + measurable knowledge transfer**. That turns PREDATOR NET from a cool visualization into something you can actually formulate as an ANN research experiment. 🧠⚔️
