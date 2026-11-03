import React, { useState, useEffect } from 'react';
import {
  Flame,
  Copy,
  Check,
  Laptop,
  Terminal,
  ShieldAlert,
  Play,
  Wifi,
  ExternalLink,
  Zap,
} from 'lucide-react';
import { fetchK6Script, simulateAttack } from '../services/api';
import { Telemetry } from '../types';

interface AttackSimulatorProps {
  telemetry: Telemetry;
  onAttackTriggered: () => void;
}

export const AttackSimulatorView: React.FC<AttackSimulatorProps> = ({
  telemetry,
  onAttackTriggered,
}) => {
  const [k6Script, setK6Script] = useState<string>('');
  const [copied, setCopied] = useState<boolean>(false);
  const [selectedAttack, setSelectedAttack] = useState<'ddos_flood' | 'port_scan' | 'syn_flood'>('ddos_flood');
  const [intensity, setIntensity] = useState<number>(60);
  const [isFiring, setIsFiring] = useState<boolean>(false);

  useEffect(() => {
    fetchK6Script().then((data) => {
      if (data && data.script) {
        setK6Script(data.script);
      }
    });
  }, []);

  const handleCopy = () => {
    navigator.clipboard.writeText(k6Script);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleFireLocal = async () => {
    setIsFiring(true);
    await simulateAttack(selectedAttack, intensity);
    onAttackTriggered();
    setTimeout(() => setIsFiring(false), 2500);
  };

  const targetIp = telemetry?.target_ip || '10.5.128.179';
  const targetPort = telemetry?.target_port || 8080;
  const targetUrl = `http://${targetIp}:${targetPort}/`;

  return (
    <div className="flex-1 flex flex-col h-full overflow-y-auto p-6 gap-6 bg-[#0B0F17]">
      {/* View Header */}
      <div>
        <h1 className="text-xl font-bold text-white flex items-center gap-2.5">
          <Flame className="w-5 h-5 text-amber-400" />
          Demonstration Hub & Penetration Attack Simulator
        </h1>
        <p className="text-xs text-slate-400 font-mono mt-0.5">
          Tools and scripts to execute real-time DDoS and penetration attacks from an external laptop using k6
        </p>
      </div>

      {/* Target Server Card */}
      <div className="bg-[#111827] border border-cyan-500/30 rounded-xl p-5 flex items-center justify-between shadow-xl">
        <div className="flex items-center gap-4">
          <div className="w-11 h-11 rounded-xl bg-cyan-500/10 text-cyan-400 flex items-center justify-center shrink-0 border border-cyan-500/30">
            <Wifi className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <span className="text-[10px] font-mono text-cyan-400 uppercase tracking-wider font-bold">
              VICTIM HTTP SERVER LISTENING ON WI-FI LAN
            </span>
            <div className="text-2xl font-mono font-bold text-white tracking-wide mt-0.5">
              {targetUrl}
            </div>
            <p className="text-xs text-slate-400 font-mono mt-1">
              External PCs on Wi-Fi (IP: 10.5.128.179) send traffic directly to this endpoint
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <a
            href={targetUrl}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-2 bg-slate-800 hover:bg-slate-700 text-slate-200 px-4 py-2 rounded-lg text-xs font-mono transition border border-slate-700"
          >
            <span>Open in Browser</span>
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>
      </div>

      {/* Two Column Layout: Teacher Instructions (Left) + Local Test Trigger (Right) */}
      <div className="grid grid-cols-12 gap-6">
        {/* Left: 3-Step Demonstration Guide */}
        <div className="col-span-7 flex flex-col gap-4">
          <div className="bg-[#111827] rounded-xl border border-slate-800 p-5 flex flex-col gap-4 shadow-xl">
            <h2 className="text-sm font-bold text-white flex items-center gap-2 font-mono">
              <Laptop className="w-4 h-4 text-emerald-400" />
              HOW TO DEMONSTRATE TO YOUR TEACHER (FROM ANOTHER PC)
            </h2>

            {/* Step 1 */}
            <div className="flex gap-3">
              <div className="w-6 h-6 rounded-full bg-emerald-500/20 text-emerald-400 font-bold text-xs flex items-center justify-center shrink-0 border border-emerald-500/40 font-mono">
                1
              </div>
              <div className="text-xs font-mono text-slate-300">
                <strong className="text-white block mb-0.5">Connect Both PCs to the Same Wi-Fi / Router</strong>
                Ensure the external attacker laptop and this host machine are on the same Wi-Fi network.
              </div>
            </div>

            {/* Step 2 */}
            <div className="flex gap-3">
              <div className="w-6 h-6 rounded-full bg-emerald-500/20 text-emerald-400 font-bold text-xs flex items-center justify-center shrink-0 border border-emerald-500/40 font-mono">
                2
              </div>
              <div className="text-xs font-mono text-slate-300">
                <strong className="text-white block mb-0.5">Install k6 on the Attacker Laptop</strong>
                Run <code className="text-cyan-300 bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800">winget install k6</code> (Windows) or <code className="text-cyan-300 bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800">brew install k6</code> (Mac).
              </div>
            </div>

            {/* Step 3 */}
            <div className="flex gap-3">
              <div className="w-6 h-6 rounded-full bg-emerald-500/20 text-emerald-400 font-bold text-xs flex items-center justify-center shrink-0 border border-emerald-500/40 font-mono">
                3
              </div>
              <div className="text-xs font-mono text-slate-300">
                <strong className="text-white block mb-0.5">Run the k6 Flood Attack Script:</strong>
                <div className="mt-1.5 bg-black/80 p-2.5 rounded-lg border border-slate-800 text-cyan-300 font-mono text-xs flex items-center justify-between">
                  <code>k6 run --vus 50 --duration 30s k6_attack_script.js</code>
                </div>
                <span className="text-[11px] text-slate-400 block mt-1">
                  Watch this app in real time: the threat meter spikes to 99%, predator neurons illuminate, and the attacker IP is automatically identified.
                </span>
              </div>
            </div>
          </div>

          {/* k6 Script Preview */}
          <div className="bg-[#111827] rounded-xl border border-slate-800 p-5 shadow-xl flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono font-bold text-white flex items-center gap-2">
                <Terminal className="w-4 h-4 text-cyan-400" />
                PRE-CONFIGURED K6 SCRIPT (k6_attack_script.js)
              </span>
              <button
                onClick={handleCopy}
                className="flex items-center gap-1.5 bg-slate-800 hover:bg-slate-700 text-xs font-mono px-3 py-1.5 rounded-lg text-slate-300 border border-slate-700 transition"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copied ? 'Copied!' : 'Copy Script'}</span>
              </button>
            </div>

            <div className="bg-black/80 rounded-lg p-3 border border-slate-800 text-cyan-300 font-mono text-[11px] overflow-x-auto max-h-48 whitespace-pre leading-relaxed">
              {k6Script || '// Loading k6 script...'}
            </div>
          </div>
        </div>

        {/* Right: Instant Local Test Attack Trigger */}
        <div className="col-span-5 flex flex-col gap-4">
          <div className="bg-[#111827] rounded-xl border border-slate-800 p-5 flex flex-col justify-between shadow-xl">
            <div>
              <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-4">
                <h3 className="text-sm font-bold text-white font-mono flex items-center gap-2">
                  <Zap className="w-4 h-4 text-amber-400" />
                  INSTANT LOCAL TEST TRIGGER
                </h3>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-500/15 text-amber-300 border border-amber-500/30 font-bold">
                  OFFLINE DEMO
                </span>
              </div>

              <p className="text-xs text-slate-400 font-mono mb-4 leading-relaxed">
                If another laptop is not available, trigger local test attack flows directly to demonstrate real-time PREDATOR detection and neuron hunting:
              </p>

              {/* Attack Type Selector */}
              <div className="space-y-2 mb-4 font-mono text-xs">
                <label className="text-slate-400 text-[11px] block">SELECT ATTACK SIGNATURE:</label>
                <div className="grid grid-cols-1 gap-2">
                  <button
                    onClick={() => setSelectedAttack('ddos_flood')}
                    className={`p-3 rounded-lg border text-left transition ${
                      selectedAttack === 'ddos_flood'
                        ? 'bg-rose-500/15 border-rose-500/40 text-white font-bold'
                        : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-white'
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <span className="text-white">Volumetric HTTP Flood (DDoS)</span>
                      <span className="text-[10px] text-rose-400">High Volume</span>
                    </div>
                    <span className="text-[10px] text-slate-400 block mt-0.5">
                      Rapid HTTP requests triggering flow_packets_per_s anomalies
                    </span>
                  </button>

                  <button
                    onClick={() => setSelectedAttack('port_scan')}
                    className={`p-3 rounded-lg border text-left transition ${
                      selectedAttack === 'port_scan'
                        ? 'bg-rose-500/15 border-rose-500/40 text-white font-bold'
                        : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-white'
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <span className="text-white">PortScan / Stealth Probe</span>
                      <span className="text-[10px] text-amber-400">Reconnaissance</span>
                    </div>
                    <span className="text-[10px] text-slate-400 block mt-0.5">
                      Sequential SYN probes across multiple TCP destination ports
                    </span>
                  </button>
                </div>
              </div>

              {/* Intensity Slider */}
              <div className="space-y-1.5 mb-5 font-mono text-xs">
                <div className="flex justify-between text-slate-300">
                  <span className="text-slate-400">ATTACK BURST DENSITY:</span>
                  <strong className="text-white">{intensity} Flows</strong>
                </div>
                <input
                  type="range"
                  min="20"
                  max="120"
                  step="10"
                  value={intensity}
                  onChange={(e) => setIntensity(Number(e.target.value))}
                  className="w-full accent-rose-500 cursor-pointer"
                />
              </div>
            </div>

            <button
              onClick={handleFireLocal}
              disabled={isFiring}
              className={`w-full py-3 rounded-xl font-mono text-xs font-bold flex items-center justify-center gap-2 transition shadow-xl ${
                isFiring
                  ? 'bg-rose-800 text-white cursor-wait'
                  : 'bg-rose-600 hover:bg-rose-500 text-white shadow-rose-600/30'
              }`}
            >
              <Play className={`w-4 h-4 ${isFiring ? 'animate-spin' : ''}`} />
              <span>{isFiring ? 'INJECTING ATTACK FLOWS...' : 'LAUNCH ATTACK SIMULATION'}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
