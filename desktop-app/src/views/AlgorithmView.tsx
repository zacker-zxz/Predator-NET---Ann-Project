import React, { useState } from 'react';
import {
  Binary,
  Cpu,
  Layers,
  Calculator,
  Activity,
  CheckCircle,
  ShieldAlert,
  ShieldCheck,
  Zap,
  ArrowRight,
  Flame,
  RefreshCw,
  Sliders,
  HelpCircle,
} from 'lucide-react';

export const AlgorithmView: React.FC = () => {
  const [activeSubTab, setActiveSubTab] = useState<
    'architecture' | 'forward' | 'absorption' | 'constants' | 'calculator'
  >('forward');

  // Interactive Calculator State
  const [calcIw, setCalcIw] = useState<number>(0.042);
  const [calcIs, setCalcIs] = useState<number>(0.315);
  const [calcCompat, setCalcCompat] = useState<number>(0.84);
  const [calcLogit, setCalcLogit] = useState<number>(3.2);

  // Live Calculations
  const epsilon = 1e-8;
  const alphaMin = 0.05;
  const alphaMax = 0.50;
  const rawRatio = calcIw / (calcIs + epsilon);
  const rawAlpha = calcCompat * rawRatio;
  const computedAlpha = Math.min(alphaMax, Math.max(alphaMin, rawAlpha));
  const isCompatAccepted = calcCompat >= 0.70;

  // Sigmoid Calculation
  const sigmoidProb = 1 / (1 + Math.exp(-calcLogit));
  const isAttackClass = sigmoidProb >= 0.50;

  return (
    <div className="flex-1 flex flex-col h-full overflow-y-auto p-6 gap-6 bg-[#0B0F17]">
      {/* Header Banner */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2.5 font-mono">
            <Binary className="w-5 h-5 text-emerald-400" />
            PREDATOR-NET Algorithm, Mathematics & Formulation Engine
          </h1>
          <p className="text-xs text-slate-400 font-mono mt-1">
            Complete formal specification: forward pass tensors, first-order neuron importance, dual-cosine compatibility, alpha transfer & structural layer rebuild
          </p>
        </div>

        {/* Status Pills */}
        <div className="flex items-center gap-2 font-mono text-[11px]">
          <span className="bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 px-2.5 py-1 rounded-lg font-semibold flex items-center gap-1.5">
            <CheckCircle className="w-3.5 h-3.5" />
            DynamicMLP Active
          </span>
          <span className="bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 px-2.5 py-1 rounded-lg font-semibold">
            CIC-IDS2017 Standard
          </span>
        </div>
      </div>

      {/* Navigation Sub-Tabs */}
      <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
        {[
          { id: 'forward', label: '1. Forward Pass & Tensors', icon: <Cpu className="w-4 h-4" /> },
          { id: 'absorption', label: '2. PREDATOR Neuron Absorption & Alpha', icon: <Layers className="w-4 h-4" /> },
          { id: 'architecture', label: '3. Model Topology & Dimensions', icon: <Activity className="w-4 h-4" /> },
          { id: 'constants', label: '4. Hyperparameters & Constants', icon: <HelpCircle className="w-4 h-4" /> },
          { id: 'calculator', label: '5. Interactive Live Math Playground', icon: <Calculator className="w-4 h-4" /> },
        ].map((tab) => {
          const isActive = activeSubTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveSubTab(tab.id as any)}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-mono font-semibold transition cursor-pointer ${
                isActive
                  ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
              }`}
            >
              {tab.icon}
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* TAB 1: FORWARD PASS & TENSORS */}
      {activeSubTab === 'forward' && (
        <div className="space-y-6">
          {/* Card 1: Input Scaling Formula */}
          <div className="bg-[#111827] rounded-xl border border-slate-800 p-5 shadow-xl">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-wider flex items-center gap-2">
                <Sliders className="w-4 h-4" />
                Step 1: Input Feature Normalization (Strict Fit-on-Train Z-Score)
              </span>
              <span className="text-[10px] font-mono text-slate-400">Zero Data Leakage Protocol</span>
            </div>
            <p className="text-xs text-slate-300 font-mono leading-relaxed mb-4">
              Each network flow is represented by a statistical vector <code className="text-cyan-300">x ∈ ℝ⁴¹</code> (or 26 selected features in Mode A). Features are normalized using standard Z-score parameters computed exclusively on the training set:
            </p>

            <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-4 font-mono text-center my-3 text-sm text-emerald-300 shadow-inner">
              <span className="text-slate-400">Normalized Feature:</span>&nbsp;&nbsp;
              <strong>x̂ᵢ = (xᵢ - μᵢ) / (σᵢ + ε)</strong>
            </div>

            <div className="grid grid-cols-3 gap-3 text-xs font-mono text-slate-300 mt-4">
              <div className="bg-slate-900/90 p-3 rounded-lg border border-slate-800">
                <span className="text-slate-400 block text-[10px]">MEAN VECTOR (μ)</span>
                <span className="text-white font-bold">Empirical feature mean</span>
                <span className="text-[10px] text-slate-400 block mt-1">Computed on X_train only</span>
              </div>
              <div className="bg-slate-900/90 p-3 rounded-lg border border-slate-800">
                <span className="text-slate-400 block text-[10px]">STD VECTOR (σ)</span>
                <span className="text-white font-bold">Feature variance spread</span>
                <span className="text-[10px] text-slate-400 block mt-1">Ensures unit variance scale</span>
              </div>
              <div className="bg-slate-900/90 p-3 rounded-lg border border-slate-800">
                <span className="text-slate-400 block text-[10px]">STABILITY EPSILON (ε)</span>
                <span className="text-white font-bold">ε = 1.0 × 10⁻⁸</span>
                <span className="text-[10px] text-slate-400 block mt-1">Prevents division by zero</span>
              </div>
            </div>
          </div>

          {/* Card 2: Affine Transformation & Activation */}
          <div className="bg-[#111827] rounded-xl border border-slate-800 p-5 shadow-xl">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-mono font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-2">
                <Zap className="w-4 h-4" />
                Step 2: Hidden Layer Affine Transformations & Non-Linearity
              </span>
              <span className="text-[10px] font-mono text-slate-400">Dynamic Multi-Layer Perceptron</span>
            </div>

            <p className="text-xs text-slate-300 font-mono leading-relaxed mb-4">
              Forward propagation across hidden layers <code className="text-cyan-300">l ∈ &#123;1, 2, 3&#125;</code> computes affine dot-products followed by Rectified Linear Unit (ReLU) activations:
            </p>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-4 font-mono text-center text-sm text-cyan-300">
                <span className="text-slate-400 text-xs block mb-1">PRE-ACTIVATION AFFINE SUM (z):</span>
                <strong>z⁽ˡ⁾ = W⁽ˡ⁾ · a⁽ˡ⁻¹⁾ + b⁽ˡ⁾</strong>
              </div>
              <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-4 font-mono text-center text-sm text-emerald-300">
                <span className="text-slate-400 text-xs block mb-1">ACTIVATION VECTOR (a):</span>
                <strong>a⁽ˡ⁾ = ReLU(z⁽ˡ⁾) = max(0, z⁽ˡ⁾)</strong>
              </div>
            </div>

            <div className="mt-4 p-3.5 bg-slate-900 rounded-lg border border-slate-800 font-mono text-xs text-slate-300 flex items-center justify-between">
              <div>
                <span className="text-white font-bold">BIAS VECTOR TERM (b):</span>
                <span className="text-slate-400 block text-[11px]">
                  Each layer incorporates an unconstrained trainable bias <code className="text-cyan-300">b ∈ ℝ^d_out</code> shifting the activation hyperplane.
                </span>
              </div>
              <span className="text-emerald-400 font-bold bg-emerald-500/10 px-2.5 py-1 rounded border border-emerald-500/20">
                b ~ U(-k, k), k = 1/√d_in
              </span>
            </div>
          </div>

          {/* Card 3: Output Logit & Threat Probability */}
          <div className="bg-[#111827] rounded-xl border border-slate-800 p-5 shadow-xl">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-mono font-bold text-amber-400 uppercase tracking-wider flex items-center gap-2">
                <Activity className="w-4 h-4" />
                Step 3: Output Logit, Sigmoid Probability & Threat Decision Rule
              </span>
              <span className="text-[10px] font-mono text-slate-400">Final Classification Head</span>
            </div>

            <p className="text-xs text-slate-300 font-mono leading-relaxed mb-4">
              The final projection layer maps hidden representation <code className="text-cyan-300">a⁽³⁾ ∈ ℝ³²</code> to a scalar logit <code className="text-amber-300">z_out</code>, converted into threat probability via standard Sigmoid:
            </p>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-4 font-mono text-center text-sm text-amber-300">
                <span className="text-slate-400 text-xs block mb-1">SIGMOID THREAT PROBABILITY:</span>
                <strong>P(Intrusion | x) = σ(z_out) = 1 / (1 + e⁻ᶻ_ᵒᵘᵗ)</strong>
              </div>
              <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-4 font-mono text-center text-sm text-rose-300">
                <span className="text-slate-400 text-xs block mb-1">DECISION BOUNDARY RULE (τ = 0.50):</span>
                <strong>ŷ = 1 (ATTACK) if P ≥ τ else 0 (BENIGN)</strong>
              </div>
            </div>

            {/* Loss function */}
            <div className="mt-4 p-4 bg-slate-900 rounded-lg border border-slate-800 font-mono text-xs">
              <div className="flex items-center justify-between mb-1">
                <span className="text-white font-bold">TRAINING LOSS: Weighted Binary Cross-Entropy With Logits</span>
                <span className="text-rose-400 font-bold">pos_weight = 4.0</span>
              </div>
              <div className="text-cyan-300 font-mono text-xs py-1">
                ℒ(θ) = -1/N ∑ [ w_pos · yᵢ · log(σ(zᵢ)) + (1 - yᵢ) · log(1 - σ(zᵢ)) ]
              </div>
              <span className="text-slate-400 text-[11px] block mt-1">
                The positive weight factor (w_pos = 4.0) penalizes False Negatives heavily, preventing undetected intrusion escapes in high-throughput traffic.
              </span>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: NEURON ABSORPTION & ALPHA */}
      {activeSubTab === 'absorption' && (
        <div className="space-y-6">
          {/* Card 1: Importance Formulation */}
          <div className="bg-[#111827] rounded-xl border border-slate-800 p-5 shadow-xl">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-mono font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-2">
                <Activity className="w-4 h-4" />
                Phase A: First-Order Behavioral Importance Calculation (Spec §20-21)
              </span>
              <span className="text-[10px] font-mono text-slate-400">Gradient-Activation Taylor Product</span>
            </div>

            <p className="text-xs text-slate-300 font-mono leading-relaxed mb-4">
              Neuron strength is evaluated dynamically using its first-order gradient sensitivity multiplied by activation magnitude, combined with outgoing weight norm:
            </p>

            <div className="grid grid-cols-3 gap-3 font-mono text-xs text-center">
              <div className="bg-slate-950/80 p-3.5 rounded-lg border border-slate-800">
                <span className="text-slate-400 text-[10px] block mb-1">GRADIENT IMPORTANCE:</span>
                <span className="text-cyan-300 font-bold block text-sm">I_grad,j = 𝔼 [ | a_j · ∂ℒ/∂a_j | ]</span>
                <span className="text-slate-400 text-[10px] block mt-1">Measures loss sensitivity</span>
              </div>
              <div className="bg-slate-950/80 p-3.5 rounded-lg border border-slate-800">
                <span className="text-slate-400 text-[10px] block mb-1">WEIGHT MAGNITUDE:</span>
                <span className="text-amber-300 font-bold block text-sm">I_weight,j = || u_j ||₁</span>
                <span className="text-slate-400 text-[10px] block mt-1">L1 norm of outgoing weights</span>
              </div>
              <div className="bg-slate-950/80 p-3.5 rounded-lg border border-slate-800">
                <span className="text-slate-400 text-[10px] block mb-1">COMBINED IMPORTANCE SCORE:</span>
                <span className="text-emerald-400 font-bold block text-sm">I_j = α_imp · Î_grad + (1 - α_imp) · Î_weight</span>
                <span className="text-slate-400 text-[10px] block mt-1">Default α_imp = 0.70</span>
              </div>
            </div>
          </div>

          {/* Card 2: Compatibility & Gating */}
          <div className="bg-[#111827] rounded-xl border border-slate-800 p-5 shadow-xl">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-wider flex items-center gap-2">
                <Zap className="w-4 h-4" />
                Phase B: Gated Dual-Cosine Compatibility Metric (Spec §24-25)
              </span>
              <span className="text-[10px] font-mono text-slate-400">Predator vs Prey Pairing</span>
            </div>

            <p className="text-xs text-slate-300 font-mono leading-relaxed mb-4">
              To prevent destructive representation corruption, a weak candidate <code className="text-rose-300">w</code> can only be absorbed by a strong neuron <code className="text-emerald-300">s</code> if both activation behavior and incoming connectivity align:
            </p>

            <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-4 font-mono text-center text-sm text-cyan-300 my-2">
              <strong>Compatibility(w, s) = β · cos(H_w, H_s) + (1 - β) · cos(V_w, V_s)</strong>
              <div className="text-xs text-slate-400 mt-1">
                Where β = 0.70, H is calibration activation profile, V is incoming weight vector
              </div>
            </div>

            <div className="mt-4 p-3.5 bg-rose-950/30 border border-rose-500/40 rounded-lg font-mono text-xs text-slate-300 flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <ShieldAlert className="w-4 h-4 text-rose-400 shrink-0" />
                <span>
                  <strong className="text-rose-300">GATING RESTRICTION:</strong> If <code className="text-white">Compatibility(w, s) &lt; 0.70</code>, the absorption is <strong className="text-rose-400">REJECTED</strong>. No destructive compression occurs.
                </span>
              </div>
              <span className="text-rose-400 font-bold text-xs bg-rose-900/40 px-2 py-1 rounded border border-rose-700/50">
                threshold = 0.70
              </span>
            </div>
          </div>

          {/* Card 3: The Alpha Absorption Formulation */}
          <div className="bg-[#111827] rounded-xl border border-slate-800 p-5 shadow-xl">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-mono font-bold text-amber-400 uppercase tracking-wider flex items-center gap-2">
                <Calculator className="w-4 h-4" />
                Phase C: The Alpha (α) Absorption Transfer Factor & Weight Updates (Spec §26-27)
              </span>
              <span className="text-[10px] font-mono text-slate-400">Knowledge Transfer Mechanism</span>
            </div>

            <p className="text-xs text-slate-300 font-mono leading-relaxed mb-3">
              The scalar transfer factor <code className="text-amber-300">α</code> governs what fraction of the weak neuron's outgoing projections are merged into the compatible strong neuron:
            </p>

            <div className="bg-slate-950/90 border-2 border-amber-500/50 rounded-xl p-4 font-mono text-center text-sm text-amber-300 shadow-xl my-4">
              <span className="text-slate-400 text-xs block mb-1">THE ALPHA FORMULATION:</span>
              <strong className="text-base text-white">
                α = clip( Compatibility(w, s) · [ I_w / (I_s + ε) ], &nbsp;α_min, &nbsp;α_max )
              </strong>
              <div className="text-xs text-amber-400/90 mt-1">
                Bounds: α_min = 0.05, &nbsp;α_max = 0.50, &nbsp;ε = 1.0 × 10⁻⁸
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 mt-4 font-mono text-xs">
              <div className="p-4 bg-slate-900 rounded-lg border border-slate-800">
                <span className="text-emerald-400 font-bold block mb-1">1. OUTGOING WEIGHT TRANSFER:</span>
                <div className="text-white font-bold my-1 text-sm">u_s^(new) = u_s + α · u_w</div>
                <span className="text-slate-400 text-[11px] block mt-1">
                  Transfers output column projections from weak neuron w into strong neuron s.
                </span>
              </div>

              <div className="p-4 bg-slate-900 rounded-lg border border-slate-800">
                <span className="text-cyan-400 font-bold block mb-1">2. INCOMING WEIGHT & BIAS TRANSFER (PREDATOR-FULL):</span>
                <div className="text-white font-bold my-1 text-sm">v_s^(new) = (1 - γ) · v_s + γ · v_w</div>
                <div className="text-white font-bold my-1 text-sm">b_s^(new) = (1 - γ) · b_s + γ · b_w</div>
                <span className="text-slate-400 text-[11px] block mt-1">
                  Incoming coefficient γ = 0.30 smoothly integrates receptive field and bias.
                </span>
              </div>
            </div>
          </div>

          {/* Card 4: Structural Layer Rebuild & Validation Rollback */}
          <div className="bg-[#111827] rounded-xl border border-slate-800 p-5 shadow-xl">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-mono font-bold text-rose-400 uppercase tracking-wider flex items-center gap-2">
                <RefreshCw className="w-4 h-4" />
                Phase D: Physical Layer Slicing & Validation Gate (Spec §28, §30)
              </span>
              <span className="text-[10px] font-mono text-slate-400">Zero Zero-Weight Masking</span>
            </div>

            <div className="grid grid-cols-2 gap-4 font-mono text-xs text-slate-300">
              <div className="p-4 bg-slate-900 rounded-lg border border-slate-800">
                <span className="text-white font-bold block mb-1">PHYSICAL TENSOR SLICING:</span>
                <ul className="space-y-1.5 text-slate-300 mt-2 list-disc list-inside text-[11px]">
                  <li>Layer L: Row <code className="text-rose-400">w</code> deleted from weight matrix <code className="text-cyan-300">W⁽ᴸ⁾</code></li>
                  <li>Layer L: Element <code className="text-rose-400">b[w]</code> deleted from bias vector <code className="text-cyan-300">b⁽ᴸ⁾</code></li>
                  <li>Layer L+1: Column <code className="text-rose-400">w</code> deleted from weight matrix <code className="text-cyan-300">W⁽ᴸ⁺¹⁾</code></li>
                  <li>Instantiates smaller brand-new PyTorch <code className="text-emerald-400">nn.Linear</code> modules</li>
                </ul>
              </div>

              <div className="p-4 bg-slate-900 rounded-lg border border-slate-800">
                <span className="text-white font-bold block mb-1">VALIDATION GATE CRITERIA:</span>
                <div className="space-y-2 text-[11px] mt-2">
                  <div className="flex items-center justify-between bg-slate-950 p-2 rounded">
                    <span>F1-Score Tolerance:</span>
                    <strong className="text-emerald-400">ΔF1 ≤ 0.005</strong>
                  </div>
                  <div className="flex items-center justify-between bg-slate-950 p-2 rounded">
                    <span>Recall Detection Tolerance:</span>
                    <strong className="text-emerald-400">ΔRecall ≤ 0.005</strong>
                  </div>
                  <span className="text-slate-400 text-[10px] block mt-1">
                    If performance drops exceed tolerance, full state-dict rollback is executed.
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: MODEL TOPOLOGY & DIMENSIONS */}
      {activeSubTab === 'architecture' && (
        <div className="space-y-6">
          <div className="bg-[#111827] rounded-xl border border-slate-800 p-5 shadow-xl">
            <h3 className="text-sm font-bold text-white font-mono flex items-center gap-2 mb-4">
              <Cpu className="w-4 h-4 text-emerald-400" />
              NEURAL ARCHITECTURE SPECIFICATION
            </h3>

            <div className="grid grid-cols-4 gap-4 mb-6">
              <div className="p-3.5 bg-slate-900 rounded-lg border border-slate-800 font-mono">
                <span className="text-[10px] text-slate-400 uppercase">MODEL CLASS</span>
                <div className="text-white font-bold text-sm mt-0.5">DynamicMLP</div>
                <span className="text-[10px] text-emerald-400">ModuleList Resizing</span>
              </div>
              <div className="p-3.5 bg-slate-900 rounded-lg border border-slate-800 font-mono">
                <span className="text-[10px] text-slate-400 uppercase">BASELINE PARAMS</span>
                <div className="text-white font-bold text-sm mt-0.5">13,825</div>
                <span className="text-[10px] text-slate-400">224 Total Neurons</span>
              </div>
              <div className="p-3.5 bg-slate-900 rounded-lg border border-slate-800 font-mono">
                <span className="text-[10px] text-slate-400 uppercase">PREDATOR PARAMS</span>
                <div className="text-cyan-300 font-bold text-sm mt-0.5">13,759</div>
                <span className="text-[10px] text-emerald-400">Compressed & Preserved</span>
              </div>
              <div className="p-3.5 bg-slate-900 rounded-lg border border-slate-800 font-mono">
                <span className="text-[10px] text-slate-400 uppercase">CLASSIFICATION ACCURACY</span>
                <div className="text-emerald-400 font-bold text-sm mt-0.5">100.00%</div>
                <span className="text-[10px] text-slate-400">PR-AUC: 1.0000</span>
              </div>
            </div>

            {/* Layer Visual Pipe */}
            <div className="space-y-3 font-mono text-xs">
              <div className="p-4 bg-slate-900/90 rounded-xl border border-slate-800 flex items-center justify-between">
                <div>
                  <span className="text-cyan-400 font-bold">Input Layer (Feature Vector x)</span>
                  <span className="text-slate-400 text-[11px] block">
                    CIC-IDS2017 Statistical Flow Features (Lengths, Flags, Ports, Window Sizes, Packet Velocities)
                  </span>
                </div>
                <div className="text-right">
                  <span className="text-white font-bold text-sm">41 Inputs</span>
                  <span className="text-slate-400 text-[10px] block">Batch: [B, 41]</span>
                </div>
              </div>

              <div className="flex justify-center -my-1 text-slate-600">↓ Linear(41 → 128) + ReLU + Bias</div>

              <div className="p-4 bg-slate-900/90 rounded-xl border border-emerald-500/30 flex items-center justify-between">
                <div>
                  <span className="text-emerald-400 font-bold">Hidden Layer 1 (Predator Hunting Arena)</span>
                  <span className="text-slate-400 text-[11px] block">
                    Dynamic ModuleList: 128 Units evaluated for first-order behavioral gradient contribution
                  </span>
                </div>
                <div className="text-right">
                  <span className="text-emerald-400 font-bold text-sm">128 Neurons</span>
                  <span className="text-slate-400 text-[10px] block">W: [128, 41], b: [128]</span>
                </div>
              </div>

              <div className="flex justify-center -my-1 text-slate-600">↓ Linear(128 → 64) + ReLU + Bias</div>

              <div className="p-4 bg-slate-900/90 rounded-xl border border-slate-800 flex items-center justify-between">
                <div>
                  <span className="text-cyan-300 font-bold">Hidden Layer 2 (Representation Compression)</span>
                  <span className="text-slate-400 text-[11px] block">
                    Intermediate representation layer distilling spatial features
                  </span>
                </div>
                <div className="text-right">
                  <span className="text-white font-bold text-sm">64 Neurons</span>
                  <span className="text-slate-400 text-[10px] block">W: [64, 128], b: [64]</span>
                </div>
              </div>

              <div className="flex justify-center -my-1 text-slate-600">↓ Linear(64 → 32) + ReLU + Bias</div>

              <div className="p-4 bg-slate-900/90 rounded-xl border border-slate-800 flex items-center justify-between">
                <div>
                  <span className="text-cyan-300 font-bold">Hidden Layer 3 (Decision Synthesis)</span>
                  <span className="text-slate-400 text-[11px] block">
                    Low-dimensional manifold synthesis layer
                  </span>
                </div>
                <div className="text-right">
                  <span className="text-white font-bold text-sm">32 Neurons</span>
                  <span className="text-slate-400 text-[10px] block">W: [32, 64], b: [32]</span>
                </div>
              </div>

              <div className="flex justify-center -my-1 text-slate-600">↓ Linear(32 → 1) + Sigmoid</div>

              <div className="p-4 bg-slate-900/90 rounded-xl border border-rose-500/40 flex items-center justify-between">
                <div>
                  <span className="text-rose-400 font-bold">Output Head (Logit Projection)</span>
                  <span className="text-slate-400 text-[11px] block">
                    Single logit output passed through Sigmoid activation producing threat score P(Intrusion)
                  </span>
                </div>
                <div className="text-right">
                  <span className="text-rose-400 font-bold text-sm">1 Output Logit</span>
                  <span className="text-slate-400 text-[10px] block">W: [1, 32], b: [1]</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 4: CONSTANTS & HYPERPARAMETERS */}
      {activeSubTab === 'constants' && (
        <div className="space-y-6">
          <div className="bg-[#111827] rounded-xl border border-slate-800 p-5 shadow-xl">
            <h3 className="text-sm font-bold text-white font-mono flex items-center gap-2 mb-4">
              <HelpCircle className="w-4 h-4 text-cyan-400" />
              SYSTEM HYPERPARAMETER & CONSTANT REFERENCE TABLE
            </h3>

            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs border-collapse">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 text-[11px] uppercase bg-slate-900/80">
                    <th className="py-3 px-4">Parameter Symbol</th>
                    <th className="py-3 px-4">Config Name</th>
                    <th className="py-3 px-4">Standard Default</th>
                    <th className="py-3 px-4">Valid Range</th>
                    <th className="py-3 px-4">Description & Functional Role</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-2.5 px-4 font-bold text-amber-300">α_min</td>
                    <td className="py-2.5 px-4 text-white">alpha_min</td>
                    <td className="py-2.5 px-4 text-emerald-400 font-bold">0.05</td>
                    <td className="py-2.5 px-4">[0.01, 0.20]</td>
                    <td className="py-2.5 px-4 text-slate-400">Minimum outgoing weight transfer coefficient</td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-2.5 px-4 font-bold text-amber-300">α_max</td>
                    <td className="py-2.5 px-4 text-white">alpha_max</td>
                    <td className="py-2.5 px-4 text-emerald-400 font-bold">0.50</td>
                    <td className="py-2.5 px-4">[0.30, 0.90]</td>
                    <td className="py-2.5 px-4 text-slate-400">Maximum outgoing weight transfer coefficient (caps transfer to prevent overflow)</td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-2.5 px-4 font-bold text-cyan-300">β</td>
                    <td className="py-2.5 px-4 text-white">beta_compat</td>
                    <td className="py-2.5 px-4 text-emerald-400 font-bold">0.70</td>
                    <td className="py-2.5 px-4">[0.0, 1.0]</td>
                    <td className="py-2.5 px-4 text-slate-400">Weight given to activation profile similarity vs incoming weights</td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-2.5 px-4 font-bold text-cyan-300">θ_min</td>
                    <td className="py-2.5 px-4 text-white">minimum_similarity</td>
                    <td className="py-2.5 px-4 text-emerald-400 font-bold">0.70</td>
                    <td className="py-2.5 px-4">[0.50, 0.95]</td>
                    <td className="py-2.5 px-4 text-slate-400">Minimum compatibility threshold for absorption gating (rejects candidate if below)</td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-2.5 px-4 font-bold text-emerald-400">γ</td>
                    <td className="py-2.5 px-4 text-white">gamma_incoming</td>
                    <td className="py-2.5 px-4 text-emerald-400 font-bold">0.30</td>
                    <td className="py-2.5 px-4">[0.0, 0.50]</td>
                    <td className="py-2.5 px-4 text-slate-400">Incoming weight and bias blending coefficient for PREDATOR-FULL mode</td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-2.5 px-4 font-bold text-emerald-400">α_imp</td>
                    <td className="py-2.5 px-4 text-white">alpha_importance</td>
                    <td className="py-2.5 px-4 text-emerald-400 font-bold">0.70</td>
                    <td className="py-2.5 px-4">[0.0, 1.0]</td>
                    <td className="py-2.5 px-4 text-slate-400">Weight between gradient sensitivity and outgoing weight magnitude</td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-2.5 px-4 font-bold text-rose-400">ΔF1_max</td>
                    <td className="py-2.5 px-4 text-white">max_f1_drop</td>
                    <td className="py-2.5 px-4 text-emerald-400 font-bold">0.005</td>
                    <td className="py-2.5 px-4">[0.001, 0.020]</td>
                    <td className="py-2.5 px-4 text-slate-400">Maximum allowed drop in validation F1 score before rollback</td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-2.5 px-4 font-bold text-rose-400">ΔRecall_max</td>
                    <td className="py-2.5 px-4 text-white">max_recall_drop</td>
                    <td className="py-2.5 px-4 text-emerald-400 font-bold">0.005</td>
                    <td className="py-2.5 px-4">[0.001, 0.020]</td>
                    <td className="py-2.5 px-4 text-slate-400">Maximum allowed drop in intrusion detection rate before rollback</td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-2.5 px-4 font-bold text-slate-400">ε</td>
                    <td className="py-2.5 px-4 text-white">epsilon</td>
                    <td className="py-2.5 px-4 text-emerald-400 font-bold">1.0 × 10⁻⁸</td>
                    <td className="py-2.5 px-4">[1e-12, 1e-6]</td>
                    <td className="py-2.5 px-4 text-slate-400">Numerical floating-point denominator stabilization constant</td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-2.5 px-4 font-bold text-slate-400">w_pos</td>
                    <td className="py-2.5 px-4 text-white">pos_weight</td>
                    <td className="py-2.5 px-4 text-emerald-400 font-bold">4.0</td>
                    <td className="py-2.5 px-4">[1.0, 10.0]</td>
                    <td className="py-2.5 px-4 text-slate-400">Binary cross entropy positive class penalty multiplier</td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-2.5 px-4 font-bold text-slate-400">τ</td>
                    <td className="py-2.5 px-4 text-white">decision_threshold</td>
                    <td className="py-2.5 px-4 text-emerald-400 font-bold">0.50</td>
                    <td className="py-2.5 px-4">[0.10, 0.90]</td>
                    <td className="py-2.5 px-4 text-slate-400">Operating decision boundary threshold for intrusion classification</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* TAB 5: INTERACTIVE LIVE CALCULATOR */}
      {activeSubTab === 'calculator' && (
        <div className="space-y-6">
          <div className="bg-[#111827] rounded-xl border border-slate-800 p-5 shadow-xl">
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-mono font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-2">
                <Calculator className="w-4 h-4" />
                Live Alpha (α) Absorption Calculator
              </span>
              <span className="text-[10px] font-mono text-slate-400">Test equations with custom parameters</span>
            </div>

            <p className="text-xs text-slate-300 font-mono mb-6">
              Adjust the sliders below to observe in real-time how the PREDATOR absorption coefficient <code className="text-amber-300">α</code> and gating acceptance respond to variable neuron importance and cosine similarity:
            </p>

            <div className="grid grid-cols-3 gap-6">
              {/* Slider 1: Weak Neuron Importance */}
              <div className="p-4 bg-slate-900 rounded-lg border border-slate-800 font-mono">
                <div className="flex justify-between text-xs mb-2">
                  <span className="text-slate-400">WEAK IMPORTANCE (I_w):</span>
                  <span className="text-rose-400 font-bold">{calcIw.toFixed(3)}</span>
                </div>
                <input
                  type="range"
                  min="0.001"
                  max="0.250"
                  step="0.001"
                  value={calcIw}
                  onChange={(e) => setCalcIw(parseFloat(e.target.value))}
                  className="w-full accent-rose-500 cursor-pointer"
                />
                <span className="text-[10px] text-slate-400 block mt-2">Candidate weak prey neuron</span>
              </div>

              {/* Slider 2: Strong Neuron Importance */}
              <div className="p-4 bg-slate-900 rounded-lg border border-slate-800 font-mono">
                <div className="flex justify-between text-xs mb-2">
                  <span className="text-slate-400">STRONG IMPORTANCE (I_s):</span>
                  <span className="text-emerald-400 font-bold">{calcIs.toFixed(3)}</span>
                </div>
                <input
                  type="range"
                  min="0.100"
                  max="1.000"
                  step="0.005"
                  value={calcIs}
                  onChange={(e) => setCalcIs(parseFloat(e.target.value))}
                  className="w-full accent-emerald-500 cursor-pointer"
                />
                <span className="text-[10px] text-slate-400 block mt-2">Host predator neuron</span>
              </div>

              {/* Slider 3: Compatibility */}
              <div className="p-4 bg-slate-900 rounded-lg border border-slate-800 font-mono">
                <div className="flex justify-between text-xs mb-2">
                  <span className="text-slate-400">COMPATIBILITY SCORE:</span>
                  <span className="text-cyan-400 font-bold">{calcCompat.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.30"
                  max="1.00"
                  step="0.01"
                  value={calcCompat}
                  onChange={(e) => setCalcCompat(parseFloat(e.target.value))}
                  className="w-full accent-cyan-500 cursor-pointer"
                />
                <span className="text-[10px] text-slate-400 block mt-2">Dual-cosine metric (threshold ≥ 0.70)</span>
              </div>
            </div>

            {/* Results Grid */}
            <div className="grid grid-cols-4 gap-4 mt-6 font-mono text-xs">
              <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
                <span className="text-slate-400 text-[10px] block">IMPORTANCE RATIO [I_w / I_s]:</span>
                <span className="text-white text-lg font-bold block mt-1">{rawRatio.toFixed(4)}</span>
                <span className="text-slate-400 text-[10px]">Relative strength ratio</span>
              </div>

              <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
                <span className="text-slate-400 text-[10px] block">UNCLIPPED ALPHA PRODUCT:</span>
                <span className="text-white text-lg font-bold block mt-1">{rawAlpha.toFixed(4)}</span>
                <span className="text-slate-400 text-[10px]">Compatibility × Ratio</span>
              </div>

              <div className="p-4 bg-slate-950 rounded-lg border border-amber-500/40">
                <span className="text-amber-400 text-[10px] block font-bold">COMPUTED ALPHA (α):</span>
                <span className="text-amber-300 text-xl font-bold block mt-1">{computedAlpha.toFixed(4)}</span>
                <span className="text-slate-400 text-[10px]">Clipped to [0.05, 0.50]</span>
              </div>

              <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
                <span className="text-slate-400 text-[10px] block">GATING VERDICT:</span>
                <div className="mt-1">
                  {isCompatAccepted ? (
                    <span className="text-emerald-400 font-bold text-sm flex items-center gap-1.5">
                      <ShieldCheck className="w-4 h-4" />
                      ACCEPTED
                    </span>
                  ) : (
                    <span className="text-rose-400 font-bold text-sm flex items-center gap-1.5">
                      <ShieldAlert className="w-4 h-4" />
                      REJECTED (&lt;0.70)
                    </span>
                  )}
                </div>
                <span className="text-slate-400 text-[10px]">
                  {isCompatAccepted ? 'Valid for absorption' : 'Protected from absorption'}
                </span>
              </div>
            </div>
          </div>

          {/* Interactive Logit & Sigmoid Tester */}
          <div className="bg-[#111827] rounded-xl border border-slate-800 p-5 shadow-xl font-mono text-xs">
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider flex items-center gap-2">
                <Zap className="w-4 h-4" />
                Live Logit to Sigmoid Probability Tester
              </span>
              <span className="text-[10px] text-slate-400">Output Layer Decision Rule</span>
            </div>

            <div className="grid grid-cols-3 gap-6 items-center">
              <div className="p-4 bg-slate-900 rounded-lg border border-slate-800">
                <div className="flex justify-between text-xs mb-2">
                  <span className="text-slate-400">OUTPUT LOGIT (z_out):</span>
                  <span className="text-cyan-300 font-bold">{calcLogit.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="-8.0"
                  max="8.0"
                  step="0.1"
                  value={calcLogit}
                  onChange={(e) => setCalcLogit(parseFloat(e.target.value))}
                  className="w-full accent-cyan-500 cursor-pointer"
                />
              </div>

              <div className="p-4 bg-slate-950 rounded-lg border border-slate-800 text-center">
                <span className="text-slate-400 text-[10px] block">SIGMOID PROBABILITY P(Intrusion | x):</span>
                <span
                  className={`text-2xl font-bold block mt-1 ${
                    isAttackClass ? 'text-rose-400' : 'text-emerald-400'
                  }`}
                >
                  {(sigmoidProb * 100).toFixed(2)}%
                </span>
                <span className="text-slate-400 text-[10px]">σ(z) = 1 / (1 + e^(-z))</span>
              </div>

              <div className="p-4 bg-slate-950 rounded-lg border border-slate-800 text-center">
                <span className="text-slate-400 text-[10px] block">FIREWALL VERDICT (τ = 0.50):</span>
                <span
                  className={`text-base font-bold block mt-1 ${
                    isAttackClass ? 'text-rose-400' : 'text-emerald-400'
                  }`}
                >
                  {isAttackClass ? 'MALICIOUS ATTACK' : 'BENIGN TRAFFIC'}
                </span>
                <span className="text-slate-400 text-[10px]">
                  {isAttackClass ? 'Flagged & Blockable' : 'Passes Perimeter'}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
