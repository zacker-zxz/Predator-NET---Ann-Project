import React, { useState, useEffect } from 'react';
import {
  Settings,
  Volume2,
  VolumeX,
  Sliders,
  Cpu,
  CheckCircle,
  Wifi,
} from 'lucide-react';
import { fetchInterfaces, setSensitivity } from '../services/api';
import { isSoundEnabled, setSoundEnabled } from '../services/sound';
import { Telemetry } from '../types';

interface SettingsProps {
  telemetry: Telemetry;
}

export const SettingsView: React.FC<SettingsProps> = ({ telemetry }) => {
  const [interfaces, setInterfaces] = useState<{ id: string; name: string }[]>([]);
  const [selectedIface, setSelectedIface] = useState<string>('default');
  const [threshold, setThreshold] = useState<number>(0.50);
  const [sound, setSound] = useState<boolean>(isSoundEnabled());
  const [savedMsg, setSavedMsg] = useState<string | null>(null);

  useEffect(() => {
    fetchInterfaces().then((data) => {
      if (Array.isArray(data)) setInterfaces(data);
    });
  }, []);

  const handleSensitivityChange = async (val: number) => {
    setThreshold(val);
    await setSensitivity(val);
    setSavedMsg('Sensitivity threshold saved.');
    setTimeout(() => setSavedMsg(null), 2500);
  };

  const handleToggleSound = () => {
    const next = !sound;
    setSound(next);
    setSoundEnabled(next);
  };

  return (
    <div className="flex-1 flex flex-col h-full overflow-y-auto p-6 gap-6 bg-[#0B0F17]">
      <div>
        <h1 className="text-xl font-bold text-white flex items-center gap-2.5">
          <Settings className="w-5 h-5 text-emerald-400" />
          Defense Engine Settings & Diagnostics
        </h1>
        <p className="text-xs text-slate-400 font-mono mt-0.5">
          Configure network sniffing adapters, sensitivity thresholds, sound alerts, and GPU acceleration
        </p>
      </div>

      {savedMsg && (
        <div className="bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 px-4 py-2.5 rounded-lg text-xs font-mono flex items-center gap-2">
          <CheckCircle className="w-4 h-4 text-emerald-400" />
          <span>{savedMsg}</span>
        </div>
      )}

      <div className="grid grid-cols-2 gap-6">
        {/* Card 1: Network Sniffer Adapter */}
        <div className="bg-[#111827] rounded-xl border border-slate-800 p-5 flex flex-col justify-between shadow-xl">
          <div>
            <h3 className="text-sm font-bold text-white font-mono flex items-center gap-2 mb-2">
              <Wifi className="w-4 h-4 text-cyan-400" />
              NETWORK CAPTURE ADAPTER
            </h3>
            <p className="text-xs text-slate-400 font-mono mb-4">
              Select the network interface to sniff packets and evaluate live web traffic.
            </p>
            <select
              value={selectedIface}
              onChange={(e) => setSelectedIface(e.target.value)}
              className="w-full bg-[#1E293B]/60 border border-slate-700 text-white rounded-lg p-2.5 text-xs font-mono focus:outline-none focus:border-emerald-500"
            >
              <option value="default">Default Gateway Interface (Auto)</option>
              {interfaces.map((iface) => (
                <option key={iface.id} value={iface.id}>
                  {iface.name}
                </option>
              ))}
            </select>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800 text-[11px] font-mono text-slate-400">
            Active: <span className="text-white font-bold">{selectedIface}</span>
          </div>
        </div>

        {/* Card 2: Sensitivity Threshold */}
        <div className="bg-[#111827] rounded-xl border border-slate-800 p-5 flex flex-col justify-between shadow-xl">
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-bold text-white font-mono flex items-center gap-2">
                <Sliders className="w-4 h-4 text-emerald-400" />
                ATTACK SENSITIVITY THRESHOLD
              </h3>
              <span className="text-xs font-mono font-bold text-emerald-400">
                {(threshold * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-xs text-slate-400 font-mono mb-4">
              Lower values flag subtle probes sooner; higher values require higher confidence before raising alerts.
            </p>
            <input
              type="range"
              min="0.30"
              max="0.95"
              step="0.05"
              value={threshold}
              onChange={(e) => handleSensitivityChange(Number(e.target.value))}
              className="w-full accent-emerald-500 cursor-pointer"
            />
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800 text-[11px] font-mono text-slate-400 flex justify-between">
            <span>Aggressive (30%)</span>
            <span>Balanced (50%)</span>
            <span>Conservative (95%)</span>
          </div>
        </div>

        {/* Card 3: Cyber Audio Alerts */}
        <div className="bg-[#111827] rounded-xl border border-slate-800 p-5 flex items-center justify-between shadow-xl">
          <div>
            <h3 className="text-sm font-bold text-white font-mono flex items-center gap-2 mb-1">
              {sound ? <Volume2 className="w-4 h-4 text-emerald-400" /> : <VolumeX className="w-4 h-4 text-slate-500" />}
              CYBER AUDIO ALERT CHIMES
            </h3>
            <p className="text-xs text-slate-400 font-mono">
              Synthesized tactical warning siren when volumetric attacks are detected.
            </p>
          </div>
          <button
            onClick={handleToggleSound}
            className={`px-4 py-2 rounded-lg text-xs font-mono font-bold transition border ${
              sound
                ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                : 'bg-slate-800 text-slate-400 border-slate-700 hover:text-white'
            }`}
          >
            {sound ? 'ENABLED' : 'MUTED'}
          </button>
        </div>

        {/* Card 4: GPU Acceleration */}
        <div className="bg-[#111827] rounded-xl border border-slate-800 p-5 flex items-center justify-between shadow-xl">
          <div>
            <h3 className="text-sm font-bold text-white font-mono flex items-center gap-2 mb-1">
              <Cpu className="w-4 h-4 text-cyan-400" />
              NVIDIA GPU ACCELERATOR
            </h3>
            <p className="text-xs text-slate-400 font-mono">
              Direct2D Canvas hardware acceleration with NVIDIA Tensor Core acceleration.
            </p>
          </div>
          <span className="px-3 py-1.5 rounded-lg text-xs font-mono font-bold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
            ENABLED
          </span>
        </div>
      </div>
    </div>
  );
};
