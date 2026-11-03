import React, { useEffect, useRef, useState } from 'react';
import { LayerActivations } from '../types';
import { Play, Pause, FastForward, Clock, Layers, ShieldCheck, ShieldAlert } from 'lucide-react';

interface NeuralCanvasProps {
  layerActivations?: LayerActivations;
  predatorNeurons?: number[];
  weakNeurons?: number[];
  huntingMode?: boolean;
  threatProbability?: number;
  packetsPerSec?: number;
}

interface SynapticCurve {
  fromX: number;
  fromY: number;
  cp1X: number;
  cp1Y: number;
  cp2X: number;
  cp2Y: number;
  toX: number;
  toY: number;
  isPredator: boolean;
  weight: number;
}

interface SignalPulse {
  curve: SynapticCurve;
  progress: number;
  speed: number;
  color: string;
  size: number;
}

interface AbsorptionTransfer {
  fromX: number;
  fromY: number;
  toX: number;
  toY: number;
  progress: number;
  speed: number;
}

const DEFAULT_PREDATORS = [4, 12, 19, 27, 35, 44, 52, 61, 73, 85, 96, 108, 114, 122];
const DEFAULT_WEAK = [1, 8, 15, 23, 31, 40, 48, 59, 68, 79, 89, 102, 118, 126];

const INPUT_FEATURE_NAMES = [
  'destination_port', 'flow_duration', 'total_fwd_packets', 'total_bwd_packets',
  'total_len_fwd', 'total_len_bwd', 'fwd_pkt_len_max', 'fwd_pkt_len_min',
  'fwd_pkt_len_mean', 'fwd_pkt_len_std', 'bwd_pkt_len_max', 'bwd_pkt_len_min',
  'bwd_pkt_len_mean', 'bwd_pkt_len_std', 'flow_bytes_per_s', 'flow_pkts_per_s',
  'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min',
  'fwd_iat_total', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max',
  'packet_len_mean', 'packet_len_std', 'packet_len_var', 'fin_flag_count',
  'syn_flag_count', 'rst_flag_count', 'psh_flag_count', 'ack_flag_count',
  'urg_flag_count', 'down_up_ratio', 'avg_pkt_size', 'subflow_fwd_pkts',
  'subflow_fwd_bytes', 'init_win_bytes_fwd', 'init_win_bytes_bwd', 'act_data_pkt_fwd',
  'min_seg_size_fwd',
];

export const NeuralSimulationCanvas: React.FC<NeuralCanvasProps> = ({
  layerActivations,
  predatorNeurons = [],
  weakNeurons = [],
  huntingMode = false,
  threatProbability = 0.0,
  packetsPerSec = 0,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [speedMultiplier, setSpeedMultiplier] = useState<number>(0.35); // Default to cinematic slow
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [hoverInfo, setHoverInfo] = useState<{ x: number; y: number; title: string; subtitle: string; val: string } | null>(null);

  const pulsesRef = useRef<SignalPulse[]>([]);
  const absorptionRef = useRef<AbsorptionTransfer[]>([]);
  const animFrameRef = useRef<number | null>(null);

  // Synchronized props ref to avoid resetting the 60fps canvas render loop on every packet
  const propsRef = useRef({
    layerActivations,
    predatorNeurons,
    weakNeurons,
    huntingMode,
    threatProbability,
    packetsPerSec,
    speedMultiplier,
    isPaused,
  });

  useEffect(() => {
    propsRef.current = {
      layerActivations,
      predatorNeurons,
      weakNeurons,
      huntingMode,
      threatProbability,
      packetsPerSec,
      speedMultiplier,
      isPaused,
    };
  }, [layerActivations, predatorNeurons, weakNeurons, huntingMode, threatProbability, packetsPerSec, speedMultiplier, isPaused]);

  // Cached node topology
  const layoutRef = useRef<{
    inputs: { x: number; y: number; id: number; name: string; act: number }[];
    layer1: { x: number; y: number; id: number; act: number; isPredator: boolean; isWeak: boolean }[];
    layer2: { x: number; y: number; id: number; act: number }[];
    layer3: { x: number; y: number; id: number; act: number }[];
    output: { x: number; y: number; act: number };
  }>({
    inputs: [],
    layer1: [],
    layer2: [],
    layer3: [],
    output: { x: 0, y: 0, act: 0 },
  });

  const updateLayout = (w: number, h: number) => {
    const padX = 75;
    const padY = 55;
    const useW = w - padX * 2;
    const useH = h - padY * 2;

    const colX = [
      padX,
      padX + useW * 0.28,
      padX + useW * 0.56,
      padX + useW * 0.78,
      padX + useW,
    ];

    const p = propsRef.current;

    // 1. Input Layer (41 feature nodes)
    const inCount = 41;
    const inputs = [];
    const inActs = p.layerActivations?.input || [];
    for (let i = 0; i < inCount; i++) {
      const y = padY + (useH / (inCount - 1)) * i;
      inputs.push({
        x: colX[0],
        y,
        id: i,
        name: INPUT_FEATURE_NAMES[i] || `feat_${i}`,
        act: inActs[i] !== undefined ? Math.abs(inActs[i]) : 0.15,
      });
    }

    // 2. Hidden Layer 1 (128 neurons: 4 columns of 32)
    const layer1 = [];
    const h1Acts = p.layerActivations?.hidden1 || [];
    const preds = (p.predatorNeurons && p.predatorNeurons.length > 0) ? p.predatorNeurons : DEFAULT_PREDATORS;
    const weaks = (p.weakNeurons && p.weakNeurons.length > 0) ? p.weakNeurons : DEFAULT_WEAK;
    const predSet = new Set(preds);
    const weakSet = new Set(weaks);

    const l1Width = useW * 0.16;
    const subCols = 4;
    const perSubCol = 32;

    for (let i = 0; i < 128; i++) {
      const c = Math.floor(i / perSubCol);
      const r = i % perSubCol;
      const x = colX[1] + (c - 1.5) * (l1Width / subCols);
      const y = padY + (useH / (perSubCol - 1)) * r;
      const act = h1Acts[i] !== undefined ? h1Acts[i] : 0.12;
      layer1.push({
        x,
        y,
        id: i,
        act,
        isPredator: predSet.has(i),
        isWeak: weakSet.has(i),
      });
    }

    // 3. Hidden Layer 2 (64 neurons: 2 columns of 32)
    const layer2 = [];
    const h2Acts = p.layerActivations?.hidden2 || [];
    for (let i = 0; i < 64; i++) {
      const c = Math.floor(i / 32);
      const r = i % 32;
      const x = colX[2] + (c - 0.5) * 36;
      const y = padY + (useH / 31) * r;
      layer2.push({
        x,
        y,
        id: i,
        act: h2Acts[i] !== undefined ? h2Acts[i] : 0.18,
      });
    }

    // 4. Hidden Layer 3 (32 neurons: 1 column)
    const layer3 = [];
    const h3Acts = p.layerActivations?.hidden3 || [];
    for (let i = 0; i < 32; i++) {
      const y = padY + (useH / 31) * i;
      layer3.push({
        x: colX[3],
        y,
        id: i,
        act: h3Acts[i] !== undefined ? h3Acts[i] : 0.22,
      });
    }

    // 5. Output Classifier Sentinel
    const output = {
      x: colX[4],
      y: h / 2,
      act: p.threatProbability,
    };

    layoutRef.current = { inputs, layer1, layer2, layer3, output };
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const parent = canvas.parentElement;
      const cssW = parent?.clientWidth || 960;
      const cssH = parent?.clientHeight || 600;
      updateLayout(cssW, cssH);
    }
  }, [layerActivations, predatorNeurons, weakNeurons, threatProbability]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const parent = canvas.parentElement;
    let cssW = parent?.clientWidth || 960;
    let cssH = parent?.clientHeight || 600;

    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    ctx.scale(dpr, dpr);

    updateLayout(cssW, cssH);

    const onResize = () => {
      if (!canvas.parentElement) return;
      cssW = canvas.parentElement.clientWidth;
      cssH = canvas.parentElement.clientHeight;
      canvas.width = cssW * dpr;
      canvas.height = cssH * dpr;
      canvas.style.width = `${cssW}px`;
      canvas.style.height = `${cssH}px`;
      ctx.scale(dpr, dpr);
      updateLayout(cssW, cssH);
    };

    window.addEventListener('resize', onResize);

    let lastT = performance.now();

    const render = (now: number) => {
      const p = propsRef.current;
      const rawDt = Math.min(Math.max((now - lastT) / 1000, 0.008), 0.033);
      lastT = now;
      const dt = p.isPaused ? 0 : rawDt * p.speedMultiplier;

      ctx.clearRect(0, 0, cssW, cssH);

      const layout = layoutRef.current;
      const isAttack = p.threatProbability > 0.5 || p.huntingMode;

      // 1. Refined Background Architecture Grid
      ctx.strokeStyle = 'rgba(30, 41, 59, 0.35)';
      ctx.lineWidth = 1;
      const gridStep = 48;
      for (let x = 0; x < cssW; x += gridStep) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, cssH);
        ctx.stroke();
      }
      for (let y = 0; y < cssH; y += gridStep) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(cssW, y);
        ctx.stroke();
      }

      // 2. Synaptic Wire Network (Bezier Curves)
      // Input -> Layer 1
      ctx.lineWidth = 0.5;
      for (let i = 0; i < layout.inputs.length; i += 3) {
        const inp = layout.inputs[i];
        for (let j = 0; j < layout.layer1.length; j += 16) {
          const l1 = layout.layer1[j];
          ctx.strokeStyle = l1.isPredator
            ? 'rgba(16, 185, 129, 0.16)'
            : 'rgba(51, 65, 85, 0.12)';
          ctx.beginPath();
          ctx.moveTo(inp.x, inp.y);
          const cpX = (inp.x + l1.x) / 2;
          ctx.bezierCurveTo(cpX, inp.y, cpX, l1.y, l1.x, l1.y);
          ctx.stroke();
        }
      }

      // Layer 1 -> Layer 2
      for (let i = 0; i < layout.layer1.length; i += 8) {
        const l1 = layout.layer1[i];
        for (let j = 0; j < layout.layer2.length; j += 8) {
          const l2 = layout.layer2[j];
          ctx.strokeStyle = l1.isPredator
            ? 'rgba(16, 185, 129, 0.22)'
            : 'rgba(51, 65, 85, 0.14)';
          ctx.beginPath();
          ctx.moveTo(l1.x, l1.y);
          const cpX = (l1.x + l2.x) / 2;
          ctx.bezierCurveTo(cpX, l1.y, cpX, l2.y, l2.x, l2.y);
          ctx.stroke();
        }
      }

      // Layer 2 -> Layer 3
      for (let i = 0; i < layout.layer2.length; i += 5) {
        const l2 = layout.layer2[i];
        for (let j = 0; j < layout.layer3.length; j += 4) {
          const l3 = layout.layer3[j];
          ctx.strokeStyle = isAttack
            ? 'rgba(244, 63, 94, 0.28)'
            : 'rgba(56, 189, 248, 0.20)';
          ctx.beginPath();
          ctx.moveTo(l2.x, l2.y);
          const cpX = (l2.x + l3.x) / 2;
          ctx.bezierCurveTo(cpX, l2.y, cpX, l3.y, l3.x, l3.y);
          ctx.stroke();
        }
      }

      // Layer 3 -> Output
      for (let i = 0; i < layout.layer3.length; i += 2) {
        const l3 = layout.layer3[i];
        ctx.strokeStyle = isAttack
          ? 'rgba(244, 63, 94, 0.40)'
          : 'rgba(16, 185, 129, 0.28)';
        ctx.lineWidth = isAttack ? 1.0 : 0.7;
        ctx.beginPath();
        ctx.moveTo(l3.x, l3.y);
        const cpX = (l3.x + layout.output.x) / 2;
        ctx.bezierCurveTo(cpX, l3.y, cpX, layout.output.y, layout.output.x, layout.output.y);
        ctx.stroke();
      }

      // 3. Steady, capped Signal Pulse Spawning (max 32 pulses, smooth 60fps)
      if (!p.isPaused && pulsesRef.current.length < 32) {
        const spawnChance = isAttack ? 0.35 : 0.16;
        if (Math.random() < spawnChance && layout.inputs.length && layout.layer1.length) {
          const inp = layout.inputs[Math.floor(Math.random() * layout.inputs.length)];
          const l1 = layout.layer1[Math.floor(Math.random() * layout.layer1.length)];
          const cpX = (inp.x + l1.x) / 2;
          pulsesRef.current.push({
            curve: {
              fromX: inp.x,
              fromY: inp.y,
              cp1X: cpX,
              cp1Y: inp.y,
              cp2X: cpX,
              cp2Y: l1.y,
              toX: l1.x,
              toY: l1.y,
              isPredator: l1.isPredator,
              weight: l1.act,
            },
            progress: 0,
            speed: 0.55 + Math.random() * 0.15,
            color: isAttack ? '#f43f5e' : '#38bdf8',
            size: isAttack ? 2.8 : 2.2,
          });
        }

        // Layer 1 -> Layer 2 pulses
        if (Math.random() < spawnChance && layout.layer1.length && layout.layer2.length) {
          const l1 = layout.layer1[Math.floor(Math.random() * layout.layer1.length)];
          const l2 = layout.layer2[Math.floor(Math.random() * layout.layer2.length)];
          const cpX = (l1.x + l2.x) / 2;
          pulsesRef.current.push({
            curve: {
              fromX: l1.x,
              fromY: l1.y,
              cp1X: cpX,
              cp1Y: l1.y,
              cp2X: cpX,
              cp2Y: l2.y,
              toX: l2.x,
              toY: l2.y,
              isPredator: l1.isPredator,
              weight: l2.act,
            },
            progress: 0,
            speed: 0.58 + Math.random() * 0.15,
            color: l1.isPredator ? '#10b981' : '#38bdf8',
            size: l1.isPredator ? 3.0 : 2.2,
          });
        }

        // Layer 3 -> Output pulses
        if (Math.random() < spawnChance && layout.layer3.length) {
          const l3 = layout.layer3[Math.floor(Math.random() * layout.layer3.length)];
          const cpX = (l3.x + layout.output.x) / 2;
          pulsesRef.current.push({
            curve: {
              fromX: l3.x,
              fromY: l3.y,
              cp1X: cpX,
              cp1Y: l3.y,
              cp2X: cpX,
              cp2Y: layout.output.y,
              toX: layout.output.x,
              toY: layout.output.y,
              isPredator: false,
              weight: layout.output.act,
            },
            progress: 0,
            speed: 0.62 + Math.random() * 0.15,
            color: isAttack ? '#f43f5e' : '#10b981',
            size: isAttack ? 3.4 : 2.6,
          });
        }
      }

      // Update & Render Signal Pulses along Bezier curves
      pulsesRef.current = pulsesRef.current.filter((pulse) => {
        if (!p.isPaused) pulse.progress += pulse.speed * dt;
        if (pulse.progress >= 1) return false;

        const t = pulse.progress;
        const c = pulse.curve;

        // Cubic Bezier interpolation: B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
        const u = 1 - t;
        const curX = u * u * u * c.fromX + 3 * u * u * t * c.cp1X + 3 * u * t * t * c.cp2X + t * t * t * c.toX;
        const curY = u * u * u * c.fromY + 3 * u * u * t * c.cp1Y + 3 * u * t * t * c.cp2Y + t * t * t * c.toY;

        ctx.fillStyle = pulse.color;
        ctx.shadowColor = pulse.color;
        ctx.shadowBlur = 6;
        ctx.beginPath();
        ctx.arc(curX, curY, pulse.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
        return true;
      });

      // 4. PREDATOR Mathematical Weight Absorption Visualization
      const predators = layout.layer1.filter((n) => n.isPredator);
      const weaks = layout.layer1.filter((n) => n.isWeak);

      if (predators.length > 0 && weaks.length > 0) {
        const transferCount = isAttack ? 5 : 2;
        for (let b = 0; b < transferCount; b++) {
          const pred = predators[(b * 3) % predators.length];
          const weak = weaks[(b * 4 + Math.floor(now * 0.0006)) % weaks.length];

          const arcAlpha = isAttack
            ? 0.32 + Math.sin(now * 0.003 + b) * 0.15
            : 0.10 + Math.sin(now * 0.001 + b) * 0.05;

          ctx.strokeStyle = `rgba(16, 185, 129, ${arcAlpha})`;
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 4]);
          ctx.beginPath();
          ctx.moveTo(weak.x, weak.y);
          ctx.lineTo(pred.x, pred.y);
          ctx.stroke();
          ctx.setLineDash([]);

          // Spawn absorption energy packet flowing from weak -> predator (capped at 10)
          if (!p.isPaused && absorptionRef.current.length < 10 && Math.random() < (isAttack ? 0.20 : 0.06)) {
            absorptionRef.current.push({
              fromX: weak.x,
              fromY: weak.y,
              toX: pred.x,
              toY: pred.y,
              progress: 0,
              speed: 0.50 + Math.random() * 0.15,
            });
          }
        }
      }

      // Render Absorption Packets
      absorptionRef.current = absorptionRef.current.filter((ap) => {
        if (!p.isPaused) ap.progress += ap.speed * dt;
        if (ap.progress >= 1) return false;

        const curX = ap.fromX + (ap.toX - ap.fromX) * ap.progress;
        const curY = ap.fromY + (ap.toY - ap.fromY) * ap.progress;

        ctx.fillStyle = '#10b981';
        ctx.shadowColor = '#10b981';
        ctx.shadowBlur = 8;
        ctx.beginPath();
        ctx.arc(curX, curY, 2.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
        return true;
      });

      // 5. Draw Layer Neurons
      // Input Layer (41 nodes)
      layout.inputs.forEach((inp) => {
        ctx.fillStyle = 'rgba(56, 189, 248, 0.7)';
        ctx.beginPath();
        ctx.arc(inp.x, inp.y, 3.2, 0, Math.PI * 2);
        ctx.fill();
      });

      // Hidden Layer 1 (Predator vs Standard vs Weak Prey)
      layout.layer1.forEach((n) => {
        const pulse = Math.sin(now * 0.002 + n.id) * 0.15 + 0.85;

        if (n.isPredator) {
          // PREDATOR NEURON: Refined Emerald Core with Soft Ring
          const r = 5.0 + n.act * 2.5 * pulse;
          ctx.fillStyle = '#10b981';
          ctx.shadowColor = '#10b981';
          ctx.shadowBlur = isAttack ? 14 : 8;
          ctx.beginPath();
          ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
          ctx.fill();
          ctx.shadowBlur = 0;

          // Outer delicate halo ring
          ctx.strokeStyle = `rgba(16, 185, 129, ${0.4 * pulse})`;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.arc(n.x, n.y, r + 3.5 * pulse, 0, Math.PI * 2);
          ctx.stroke();
        } else if (n.isWeak) {
          // WEAK / PRUNED PREY: Muted Charcoal Slate
          ctx.fillStyle = 'rgba(51, 65, 85, 0.5)';
          ctx.beginPath();
          ctx.arc(n.x, n.y, 2.4, 0, Math.PI * 2);
          ctx.fill();
        } else {
          // STANDARD HIDDEN NEURON: Clean Muted Titanium
          ctx.fillStyle = `rgba(100, 116, 139, ${0.35 + n.act * 0.4})`;
          ctx.beginPath();
          ctx.arc(n.x, n.y, 3.0, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      // Hidden Layer 2
      layout.layer2.forEach((n) => {
        ctx.fillStyle = `rgba(56, 189, 248, ${0.35 + n.act * 0.4})`;
        ctx.beginPath();
        ctx.arc(n.x, n.y, 3.2, 0, Math.PI * 2);
        ctx.fill();
      });

      // Hidden Layer 3
      layout.layer3.forEach((n) => {
        ctx.fillStyle = isAttack ? 'rgba(244, 63, 94, 0.85)' : 'rgba(16, 185, 129, 0.75)';
        ctx.beginPath();
        ctx.arc(n.x, n.y, 3.6, 0, Math.PI * 2);
        ctx.fill();
      });

      // Output Classifier Sentinel Node
      const outRadius = 14 + (isAttack ? Math.sin(now * 0.006) * 3 : 0);
      const outColor = isAttack ? '#f43f5e' : '#10b981';
      ctx.shadowColor = outColor;
      ctx.shadowBlur = isAttack ? 22 : 12;
      ctx.fillStyle = outColor;
      ctx.beginPath();
      ctx.arc(layout.output.x, layout.output.y, outRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;

      // Outer Sentinel Ring
      ctx.strokeStyle = isAttack ? 'rgba(244, 63, 94, 0.6)' : 'rgba(16, 185, 129, 0.4)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(layout.output.x, layout.output.y, outRadius + 6, 0, Math.PI * 2);
      ctx.stroke();

      animFrameRef.current = requestAnimationFrame(render);
    };

    animFrameRef.current = requestAnimationFrame(render);

    // Interactive Node Mouse Move
    const handleMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const layout = layoutRef.current;

      // Check input features
      for (const inp of layout.inputs) {
        if (Math.hypot(inp.x - mouseX, inp.y - mouseY) < 8) {
          setHoverInfo({
            x: inp.x,
            y: inp.y,
            title: `Feature #${inp.id}`,
            subtitle: inp.name,
            val: `Activation: ${inp.act.toFixed(4)}`,
          });
          return;
        }
      }

      // Check hidden layer 1
      for (const n of layout.layer1) {
        if (Math.hypot(n.x - mouseX, n.y - mouseY) < 9) {
          setHoverInfo({
            x: n.x,
            y: n.y,
            title: `Hidden-1 Neuron #${n.id}`,
            subtitle: n.isPredator
              ? 'Predator Hunter (High Saliency)'
              : n.isWeak
              ? 'Weak Prey (Weight Absorption Target)'
              : 'Standard Filter Neuron',
            val: `Weight Energy: ${(n.act * 100).toFixed(1)}%`,
          });
          return;
        }
      }

      // Check output
      if (Math.hypot(layout.output.x - mouseX, layout.output.y - mouseY) < 18) {
        const p = propsRef.current;
        const isAtt = p.threatProbability > 0.5;
        const posX = layout.output.x > 680 ? layout.output.x - 225 : layout.output.x;
        setHoverInfo({
          x: posX,
          y: layout.output.y,
          title: 'Output Classifier Sentinel',
          subtitle: isAtt ? 'ATTACK DETECTED' : 'PERIMETER SECURE',
          val: `Threat Probability: ${(p.threatProbability * 100).toFixed(1)}%`,
        });
        return;
      }

      setHoverInfo(null);
    };

    canvas.addEventListener('mousemove', handleMouseMove);

    return () => {
      window.removeEventListener('resize', onResize);
      canvas.removeEventListener('mousemove', handleMouseMove);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, []);

  return (
    <div className="flex flex-col w-full h-full bg-[#0B0F17] rounded-xl border border-slate-800 overflow-hidden shadow-2xl">
      {/* Top Header Bar - Dedicated toolbar above canvas (NEVER overlaps neurons) */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-[#111827] border-b border-slate-800 shrink-0">
        {/* Left: Legend */}
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_8px_#10b981]" />
            <span className="text-slate-200 font-semibold text-[11px]">Predator Hunter Neurons</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-slate-600" />
            <span className="text-slate-400 text-[11px]">Pruned Prey Neurons</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400" />
            <span className="text-slate-300 text-[11px]">Input Statistical Features</span>
          </div>
          {huntingMode && (
            <div className="flex items-center gap-2 text-rose-400 font-bold text-[11px] animate-pulse">
              <span className="w-2 h-2 rounded-full bg-rose-500" />
              <span>PREDATOR ABSORPTION ENGAGED</span>
            </div>
          )}
        </div>

        {/* Right: Mathematical Equation HUD + Playback Speed Controls */}
        <div className="flex items-center gap-3">
          {/* Mathematical Absorption Badge */}
          <div className="bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-800 text-[11px] font-mono text-emerald-400 shadow-sm">
            <code>w_predator ← w_predator + α·w_prey (α=0.15)</code>
          </div>

          {/* Speed Presets */}
          <div className="flex items-center bg-slate-900 rounded-lg border border-slate-800 p-0.5 text-xs font-mono">
            <button
              onClick={() => {
                const next = !isPaused;
                setIsPaused(next);
                propsRef.current.isPaused = next;
              }}
              className={`p-1.5 rounded transition ${
                isPaused ? 'bg-amber-500/20 text-amber-300' : 'text-slate-400 hover:text-white'
              }`}
              title={isPaused ? 'Resume' : 'Pause'}
            >
              {isPaused ? <Play className="w-3.5 h-3.5 text-amber-300" /> : <Pause className="w-3.5 h-3.5" />}
            </button>

            <button
              onClick={() => {
                setSpeedMultiplier(0.25);
                setIsPaused(false);
                propsRef.current.speedMultiplier = 0.25;
                propsRef.current.isPaused = false;
              }}
              className={`px-2.5 py-1 rounded text-[11px] font-semibold transition ${
                speedMultiplier === 0.25 && !isPaused
                  ? 'bg-emerald-500/20 text-emerald-400'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              0.25x Slow
            </button>

            <button
              onClick={() => {
                setSpeedMultiplier(0.5);
                setIsPaused(false);
                propsRef.current.speedMultiplier = 0.5;
                propsRef.current.isPaused = false;
              }}
              className={`px-2.5 py-1 rounded text-[11px] font-semibold transition ${
                speedMultiplier === 0.5 && !isPaused
                  ? 'bg-emerald-500/20 text-emerald-400'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              0.5x
            </button>

            <button
              onClick={() => {
                setSpeedMultiplier(1.0);
                setIsPaused(false);
                propsRef.current.speedMultiplier = 1.0;
                propsRef.current.isPaused = false;
              }}
              className={`px-2.5 py-1 rounded text-[11px] font-semibold transition ${
                speedMultiplier === 1.0 && !isPaused
                  ? 'bg-emerald-500/20 text-emerald-400'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              1.0x Realtime
            </button>
          </div>
        </div>
      </div>

      {/* Main Canvas Container - Unrestricted full view */}
      <div className="relative flex-1 w-full h-full min-h-[360px] overflow-hidden">
        {/* Layer Bottom Labels */}
        <div className="absolute bottom-2 left-0 right-0 z-10 flex justify-between px-10 text-[10px] font-mono text-slate-500 pointer-events-none uppercase tracking-wider">
          <span>INPUT (41 FEATURES)</span>
          <span>HIDDEN 1 (128 NEURONS)</span>
          <span>HIDDEN 2 (64 NEURONS)</span>
          <span>HIDDEN 3 (32 NEURONS)</span>
          <span>OUTPUT CLASSIFIER LOGIT</span>
        </div>

        {/* Floating Hover Tooltip with Boundary Overflow Prevention */}
        {hoverInfo && (
          <div
            className="absolute z-30 pointer-events-none bg-[#0E131F]/95 text-white border border-slate-700 px-3.5 py-2.5 rounded-lg text-xs font-mono shadow-2xl backdrop-blur-md min-w-[200px] whitespace-nowrap"
            style={{
              left: hoverInfo.x > 680 ? hoverInfo.x - 225 : hoverInfo.x + 16,
              top: Math.max(hoverInfo.y - 30, 10),
            }}
          >
            <div className="font-bold text-emerald-400">{hoverInfo.title}</div>
            <div className="text-[11px] text-slate-300 mt-0.5">{hoverInfo.subtitle}</div>
            <div className="text-[11px] text-cyan-300 mt-1 font-semibold">{hoverInfo.val}</div>
          </div>
        )}

        {/* High-DPI GPU Canvas */}
        <canvas ref={canvasRef} className="w-full h-full block cursor-crosshair" />
      </div>
    </div>
  );
};

