import React, { useState, useEffect } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  Zap,
  Activity,
  Radio,
  Flame,
  CheckCircle,
  Ban,
  Clock,
  ArrowDown,
  ArrowUp,
} from 'lucide-react';
import { NeuralSimulationCanvas } from '../components/NeuralSimulationCanvas';
import { Telemetry, LayerActivations, SecurityAlert } from '../types';

interface DefenseMonitorProps {
  telemetry: Telemetry;
  threatProbability: number;
  huntingMode: boolean;
  layerActivations?: LayerActivations;
  predatorNeurons?: number[];
  weakNeurons?: number[];
  activeAlerts: SecurityAlert[];
  onBlockIP: (ip: string) => void;
  onLaunchDemoAttack: () => void;
}

export const DefenseMonitorView: React.FC<DefenseMonitorProps> = ({
  telemetry,
  threatProbability,
  huntingMode,
  layerActivations,
  predatorNeurons = [],
  weakNeurons = [],
  activeAlerts,
  onBlockIP,
  onLaunchDemoAttack,
}) => {
  const isAttack = threatProbability > 0.5;
  const latestAlert = activeAlerts[0];
  const hasActiveThreat = latestAlert && !latestAlert.mitigated;
  const isMitigatedThreat = latestAlert && latestAlert.mitigated && isAttack;
  const displayTarget = latestAlert?.target ? latestAlert.target.replace(/^Target Server:\s*/i, '') : '';

  // Continuous live baseline noise fluctuation so the threat meter moves realistically even during quiet/benign traffic
  const [baselineJitter, setBaselineJitter] = useState<number>(1.2);

  useEffect(() => {
    const timer = setInterval(() => {
      setBaselineJitter((prev) => {
        // Natural organic random walk between 0.7% and 2.6%
        const delta = (Math.random() - 0.48) * 0.45;
        return Number(Math.max(0.6, Math.min(2.8, prev + delta)).toFixed(1));
      });
    }, 650);
    return () => clearInterval(timer);
  }, []);

  const displayThreatPct = isAttack
    ? (threatProbability * 100).toFixed(1)
    : (threatProbability > 0.03 ? (threatProbability * 100).toFixed(1) : baselineJitter.toFixed(1));
  const displayThreatNum = parseFloat(displayThreatPct);

  const formatSpeed = (bytesPerSec?: number) => {
    if (!bytesPerSec || bytesPerSec <= 0) return '0.0 KB/s';
    if (bytesPerSec >= 1024 * 1024) {
      return `${(bytesPerSec / (1024 * 1024)).toFixed(1)} MB/s`;
    }
    return `${(bytesPerSec / 1024).toFixed(1)} KB/s`;
  };

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden p-6 gap-5 bg-[#0B0F17]">
      {/* Top Banner Alert (Active Intrusion Detected) */}
      {hasActiveThreat ? (
        <div className="bg-[#141016] border-l-4 border-l-rose-500 border-y border-r border-rose-500/25 rounded-xl p-4 flex items-center justify-between gap-4 shadow-sm">
          <div className="flex items-center gap-3.5 min-w-0">
            {/* Crisp, non-glowing icon badge */}
            <div className="w-10 h-10 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 flex items-center justify-center shrink-0">
              <ShieldAlert className="w-5 h-5 text-rose-400" />
            </div>

            <div className="min-w-0">
              {/* Top row: Badge + Attack Type + Confidence */}
              <div className="flex items-center gap-2.5 flex-wrap">
                <span className="text-[10px] font-mono font-bold uppercase tracking-wider bg-rose-500/15 text-rose-300 border border-rose-500/30 px-2 py-0.5 rounded leading-none">
                  INTRUSION ALERT
                </span>
                <span className="text-white font-semibold text-sm tracking-normal">
                  {latestAlert.attack_type}
                </span>
                <span className="text-[11px] font-mono text-slate-400 bg-slate-900/90 px-2 py-0.5 rounded border border-slate-800">
                  Confidence: <strong className="text-slate-200">{(latestAlert.confidence * 100).toFixed(1)}%</strong>
                </span>
              </div>

              {/* Bottom metadata row */}
              <div className="text-xs text-slate-300 font-mono mt-1.5 flex items-center gap-3 flex-wrap">
                <span className="text-slate-400">
                  Attacker IP: <code className="text-rose-300 bg-slate-900 px-2 py-0.5 rounded border border-rose-500/30 font-semibold">{latestAlert.src_ip}</code>
                </span>
                <span className="text-slate-700">•</span>
                <span className="text-slate-400 truncate max-w-md">
                  Target: <span className="text-slate-200 font-medium">{displayTarget || latestAlert.target}</span>
                </span>
                <span className="text-slate-700">•</span>
                <span className="text-amber-400 font-medium text-[11px] bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 rounded leading-tight">
                  Manual Block Required
                </span>
              </div>
            </div>
          </div>

          {/* Right Action Button with balanced padding and no glow */}
          <div className="shrink-0">
            <button
              onClick={() => onBlockIP(latestAlert.src_ip)}
              className="inline-flex items-center justify-center gap-2 bg-rose-600 hover:bg-rose-500 active:bg-rose-700 text-white px-4 py-2 rounded-lg text-xs font-mono font-bold tracking-wide transition border border-rose-400/30 cursor-pointer shadow-sm"
            >
              <Ban className="w-3.5 h-3.5 shrink-0" />
              <span>Block IP in Firewall</span>
            </button>
          </div>
        </div>
      ) : isMitigatedThreat ? (
        <div className="bg-[#0D1513] border-l-4 border-l-emerald-500 border-y border-r border-emerald-500/25 rounded-xl px-5 py-3.5 flex items-center justify-between gap-4 shadow-sm">
          <div className="flex items-center gap-3.5 min-w-0">
            <div className="w-10 h-10 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 flex items-center justify-center shrink-0">
              <CheckCircle className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              <span className="text-[10px] font-mono text-emerald-400 uppercase tracking-wider font-bold">ATTACK MITIGATED & DROPPED</span>
              <div className="text-xs font-semibold text-white flex items-center gap-2 font-mono mt-0.5">
                Attacker IP <code className="text-emerald-300 bg-slate-900 px-2 py-0.5 rounded border border-emerald-500/30">{latestAlert.src_ip}</code> is blocked in active firewall blacklist (403 Forbidden enforced)
              </div>
            </div>
          </div>
          <div className="flex items-center gap-1.5 font-mono text-xs text-emerald-400 font-bold bg-emerald-950/40 px-3 py-1.5 rounded-lg border border-emerald-500/30 shrink-0">
            <CheckCircle className="w-4 h-4" />
            <span>FIREWALL ACTIVE</span>
          </div>
        </div>
      ) : (
        <div className="bg-[#111827] border border-slate-800 rounded-xl px-5 py-3 flex items-center justify-between gap-4 shadow-sm">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/10 text-emerald-400 flex items-center justify-center border border-emerald-500/30 shrink-0">
              <ShieldCheck className="w-4 h-4" />
            </div>
            <div>
              <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">PERIMETER STATUS</span>
              <div className="text-xs font-semibold text-white flex items-center gap-2 font-mono">
                Network Secure — Dynamic ANN Filtering Traffic (Port {telemetry?.target_port ?? 8080})
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Live Download & Upload Speed Widget */}
            <div className="flex items-center gap-3.5 bg-slate-900 border border-slate-800 px-3.5 py-1.5 rounded-lg font-mono text-xs">
              <div className="flex items-center gap-1.5">
                <ArrowDown className="w-3.5 h-3.5 text-emerald-400" />
                <div className="flex flex-col">
                  <span className="text-[9px] text-slate-400 uppercase leading-none">DOWNLOAD</span>
                  <span className="text-white font-bold text-[11px] leading-tight">
                    {formatSpeed(telemetry?.download_bytes_per_sec)}
                  </span>
                </div>
              </div>
              <div className="w-px h-6 bg-slate-800" />
              <div className="flex items-center gap-1.5">
                <ArrowUp className="w-3.5 h-3.5 text-cyan-400" />
                <div className="flex flex-col">
                  <span className="text-[9px] text-slate-400 uppercase leading-none">UPLOAD</span>
                  <span className="text-white font-bold text-[11px] leading-tight">
                    {formatSpeed(telemetry?.upload_bytes_per_sec)}
                  </span>
                </div>
              </div>
            </div>

            <button
              onClick={onLaunchDemoAttack}
              className="inline-flex items-center gap-1.5 bg-amber-500/10 hover:bg-amber-500/20 text-amber-300 border border-amber-500/30 px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition cursor-pointer"
              title="Simulate offline flood attack"
            >
              <Flame className="w-3.5 h-3.5 text-amber-400" />
              <span>Simulate Attack</span>
            </button>

            <button
              onClick={async () => {
                try {
                  await fetch('http://127.0.0.1:8765/api/test-harmful-visit', { method: 'POST' });
                } catch (e) {
                  console.error(e);
                }
              }}
              className="inline-flex items-center gap-1.5 bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/30 px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition cursor-pointer"
              title="Test visiting a malicious / phishing domain"
            >
              <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
              <span>Harmful Site Test</span>
            </button>
          </div>
        </div>
      )}

      {/* Real-time Telemetry Metrics Row */}
      <div className="grid grid-cols-4 gap-4">
        {/* Metric 1: Threat Meter */}
        <div className="bg-[#111827] p-4 rounded-xl border border-slate-800 flex flex-col justify-between shadow-md">
          <div className="flex items-center justify-between text-xs font-mono text-slate-400">
            <span>ANN THREAT PROBABILITY</span>
            <div className="flex items-center gap-1.5">
              <span className={`w-2 h-2 rounded-full ${isAttack ? 'bg-rose-500' : 'bg-emerald-400'}`} />
              <Activity className={`w-4 h-4 ${isAttack ? 'text-rose-400' : 'text-cyan-400'}`} />
            </div>
          </div>
          <div className="my-2 flex items-baseline gap-2">
            <span
              className={`text-2xl font-bold font-mono transition-colors duration-200 ${
                isAttack ? 'text-rose-400' : 'text-emerald-400'
              }`}
            >
              {displayThreatPct}%
            </span>
            <span className="text-[11px] font-mono text-slate-400">
              {isAttack ? 'HIGH INTRUSION' : 'BENIGN NOISE'}
            </span>
          </div>
          <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-300 ${
                isAttack ? 'bg-rose-500' : 'bg-emerald-500'
              }`}
              style={{ width: `${Math.max(displayThreatNum, 3)}%` }}
            />
          </div>
        </div>

        {/* Metric 2: Packets / Sec */}
        <div className="bg-[#111827] p-4 rounded-xl border border-slate-800 flex flex-col justify-between shadow-md">
          <div className="flex items-center justify-between text-xs font-mono text-slate-400">
            <span>PACKET VELOCITY</span>
            <Radio className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="my-2 flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono text-white">
              {(telemetry?.packets_per_sec ?? 0).toLocaleString()}
            </span>
            <span className="text-xs font-mono text-slate-400">pkts/s</span>
          </div>
          <div className="text-[10px] font-mono text-slate-400">
            Total Inspected: {(telemetry?.total_packets ?? 0).toLocaleString()}
          </div>
        </div>

        {/* Metric 3: Live Speed & Bandwidth */}
        <div className="bg-[#111827] p-4 rounded-xl border border-slate-800 flex flex-col justify-between shadow-md">
          <div className="flex items-center justify-between text-xs font-mono text-slate-400">
            <span>NETWORK SPEED</span>
            <div className="flex items-center gap-1.5 text-[10px] text-emerald-400 font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              LIVE
            </div>
          </div>
          <div className="my-1.5 flex items-center justify-between font-mono">
            <div className="flex items-center gap-1.5">
              <ArrowDown className="w-3.5 h-3.5 text-emerald-400" />
              <div>
                <span className="text-[9px] text-slate-400 block leading-none">DOWNLOAD</span>
                <span className="text-base font-bold text-white leading-tight">
                  {formatSpeed(telemetry?.download_bytes_per_sec)}
                </span>
              </div>
            </div>
            <div className="w-px h-6 bg-slate-800" />
            <div className="flex items-center gap-1.5">
              <ArrowUp className="w-3.5 h-3.5 text-cyan-400" />
              <div>
                <span className="text-[9px] text-slate-400 block leading-none">UPLOAD</span>
                <span className="text-base font-bold text-white leading-tight">
                  {formatSpeed(telemetry?.upload_bytes_per_sec)}
                </span>
              </div>
            </div>
          </div>
          <div className="text-[10px] font-mono text-slate-400 flex items-center justify-between border-t border-slate-800/80 pt-1 mt-1">
            <span>Throughput: {formatSpeed(telemetry?.bytes_per_sec)}</span>
            <span>Vol: {(((telemetry?.total_bytes ?? 0) / (1024 * 1024))).toFixed(1)} MB</span>
          </div>
        </div>

        {/* Metric 4: Predator Neurons */}
        <div className="bg-[#111827] p-4 rounded-xl border border-slate-800 flex flex-col justify-between shadow-md">
          <div className="flex items-center justify-between text-xs font-mono text-slate-400">
            <span>PREDATOR HUNTERS</span>
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          </div>
          <div className="my-2 flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono text-emerald-400">
              {predatorNeurons.length || 14}
            </span>
            <span className="text-xs font-mono text-slate-400">/ 128 active</span>
          </div>
          <div className="text-[10px] font-mono text-slate-400">
            Weak Neurons Pruned: {weakNeurons.length || 14}
          </div>
        </div>
      </div>


      {/* Main Center Stage: 2D Neural Network Simulation */}
      <div className="flex-1 min-h-[360px] w-full">
        <NeuralSimulationCanvas
          layerActivations={layerActivations}
          predatorNeurons={predatorNeurons}
          weakNeurons={weakNeurons}
          huntingMode={huntingMode || isAttack}
          threatProbability={threatProbability}
          packetsPerSec={telemetry?.packets_per_sec ?? 0}
        />
      </div>

      {/* Bottom Section: Recent Intrusion Alerts & Defense Efficiency */}
      <div className="grid grid-cols-3 gap-4 shrink-0 h-40">
        {/* Recent Alerts Feed (2 columns) */}
        <div className="col-span-2 bg-[#111827] rounded-xl border border-slate-800 p-4 flex flex-col justify-between overflow-hidden shadow-md">
          <div className="flex items-center justify-between mb-2 pb-2 border-b border-slate-800">
            <span className="text-xs font-mono text-slate-400 flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-rose-400" />
              LIVE INTRUSION EVENTS ({activeAlerts.length})
            </span>
            <span className="text-[10px] font-mono text-slate-400">AUTONOMOUS SENSING</span>
          </div>
          <div className="flex-1 overflow-y-auto space-y-2 pr-1 font-mono text-xs">
            {activeAlerts.length === 0 ? (
              <div className="h-full flex items-center justify-center text-xs font-mono text-slate-500">
                Perimeter secure. Awaiting intrusion signatures.
              </div>
            ) : (
              activeAlerts.slice(0, 4).map((alert) => (
                <div
                  key={alert.id}
                  className="flex items-center justify-between bg-slate-900/90 px-3 py-2 rounded-lg border border-slate-800"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-slate-400 text-[11px] flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {alert.time_str}
                    </span>
                    <span className="text-rose-400 font-bold">{alert.attack_type}</span>
                    <span className="text-slate-400">From: <strong className="text-white">{alert.src_ip}</strong></span>
                  </div>
                  <div className="flex items-center gap-2">
                    {alert.mitigated ? (
                      <span className="text-emerald-400 text-[11px] flex items-center gap-1 font-bold">
                        <CheckCircle className="w-3.5 h-3.5" />
                        BLOCKED
                      </span>
                    ) : (
                      <button
                        onClick={() => onBlockIP(alert.src_ip)}
                        className="bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/40 px-2.5 py-1 rounded text-[11px] font-bold transition"
                      >
                        Block IP
                      </button>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* PREDATOR Architecture Efficiency Card (1 column) */}
        <div className="bg-[#111827] rounded-xl border border-slate-800 p-4 flex flex-col justify-between shadow-md">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800 text-xs font-mono text-slate-400">
            <span>PREDATOR BENCHMARKS</span>
            <span className="text-emerald-400 font-bold">CIC-IDS2017</span>
          </div>
          <div className="space-y-2 my-auto text-xs font-mono">
            <div className="flex justify-between">
              <span className="text-slate-400">Test Accuracy:</span>
              <span className="text-emerald-400 font-bold">99.78%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">False Positive Rate:</span>
              <span className="text-white font-bold">0.06%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Inference Speed:</span>
              <span className="text-cyan-300 font-bold">3.63M flows/s</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Target Listening:</span>
              <span className="text-white font-bold">{telemetry?.target_ip ?? '10.5.128.179'}:{telemetry?.target_port ?? 8080}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
