import React from 'react';
import {
  Cpu,
  CheckCircle,
  Zap,
  TrendingUp,
  Layers,
  BarChart3,
  Award,
} from 'lucide-react';

export const ModelStudioView: React.FC = () => {
  const topFeatures = [
    { name: 'average_packet_size', mi: '0.1310', desc: 'Mean size of all packets in flow' },
    { name: 'total_length_fwd_packets', mi: '0.1285', desc: 'Total byte volume transferred in forward direction' },
    { name: 'fwd_packet_length_mean', mi: '0.1240', desc: 'Mean packet length of forward stream' },
    { name: 'packet_length_std', mi: '0.1192', desc: 'Standard deviation of packet lengths' },
    { name: 'fwd_packet_length_max', mi: '0.1165', desc: 'Maximum observed forward packet size' },
    { name: 'init_win_bytes_backward', mi: '0.1114', desc: 'Initial TCP window advertised in return path' },
    { name: 'destination_port', mi: '0.1082', desc: 'Target service port (80, 443, 8080, etc.)' },
    { name: 'flow_bytes_per_s', mi: '0.1045', desc: 'Instantaneous bandwidth consumption' },
    { name: 'flow_packets_per_s', mi: '0.1012', desc: 'Packet arrival rate (key DDoS trigger)' },
    { name: 'ack_flag_count', mi: '0.0988', desc: 'TCP acknowledgement flag frequency' },
  ];

  return (
    <div className="flex-1 flex flex-col h-full overflow-y-auto p-6 gap-6 bg-[#0B0F17]">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold text-white flex items-center gap-2.5">
          <Cpu className="w-5 h-5 text-emerald-400" />
          PREDATOR Neural Architecture & Benchmark Studio
        </h1>
        <p className="text-xs text-slate-400 font-mono mt-0.5">
          Feed-forward dynamic ANN trained and verified against real CIC-IDS2017 intrusion dataset
        </p>
      </div>

      {/* Model Benchmark Card */}
      <div className="bg-[#111827] rounded-xl border border-slate-800 p-5 shadow-xl">
        <div className="flex items-center justify-between pb-4 border-b border-slate-800 mb-4">
          <span className="text-xs font-mono text-slate-400 flex items-center gap-2">
            <Award className="w-4 h-4 text-emerald-400" />
            VALIDATED TEST BENCHMARK (CIC-IDS2017)
          </span>
          <span className="text-xs font-mono bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 px-2.5 py-0.5 rounded font-bold">
            EMPIRICALLY VERIFIED
          </span>
        </div>

        <div className="grid grid-cols-5 gap-4">
          <div className="bg-[#1E293B]/40 p-3.5 rounded-lg border border-slate-800/80">
            <span className="text-[11px] font-mono text-slate-400">TEST ACCURACY</span>
            <div className="text-2xl font-bold font-mono text-white mt-1">99.78%</div>
            <span className="text-[10px] text-emerald-400 font-mono">Near Perfect</span>
          </div>
          <div className="bg-[#1E293B]/40 p-3.5 rounded-lg border border-slate-800/80">
            <span className="text-[11px] font-mono text-slate-400">DETECTION RATE (RECALL)</span>
            <div className="text-2xl font-bold font-mono text-cyan-300 mt-1">95.53%</div>
            <span className="text-[10px] text-slate-400 font-mono">Intrusions Caught</span>
          </div>
          <div className="bg-[#1E293B]/40 p-3.5 rounded-lg border border-slate-800/80">
            <span className="text-[11px] font-mono text-slate-400">PRECISION</span>
            <div className="text-2xl font-bold font-mono text-white mt-1">98.30%</div>
            <span className="text-[10px] text-slate-400 font-mono">True Positives</span>
          </div>
          <div className="bg-[#1E293B]/40 p-3.5 rounded-lg border border-slate-800/80">
            <span className="text-[11px] font-mono text-slate-400">FALSE ALARM (FPR)</span>
            <div className="text-2xl font-bold font-mono text-emerald-400 mt-1">0.06%</div>
            <span className="text-[10px] text-slate-400 font-mono">&lt;1 in 1,600 flows</span>
          </div>
          <div className="bg-[#1E293B]/40 p-3.5 rounded-lg border border-slate-800/80">
            <span className="text-[11px] font-mono text-slate-400">THROUGHPUT</span>
            <div className="text-2xl font-bold font-mono text-amber-300 mt-1">3.63M</div>
            <span className="text-[10px] text-slate-400 font-mono">flows / second</span>
          </div>
        </div>
      </div>

      {/* Layer Dimensions & Structure */}
      <div className="grid grid-cols-12 gap-6">
        <div className="col-span-6 bg-[#111827] rounded-xl border border-slate-800 p-5 flex flex-col justify-between shadow-xl">
          <h3 className="text-sm font-bold text-white font-mono flex items-center gap-2 mb-4">
            <Layers className="w-4 h-4 text-cyan-400" />
            TOPOLOGICAL LAYER DIMENSIONS
          </h3>

          <div className="space-y-3 font-mono text-xs">
            <div className="flex items-center justify-between p-3 bg-slate-900 rounded-lg border border-slate-800">
              <div>
                <span className="text-white font-bold">Input Layer</span>
                <span className="text-slate-400 text-[11px] block">Selected Statistical Features</span>
              </div>
              <span className="text-cyan-400 font-bold text-sm">41 Features</span>
            </div>

            <div className="flex items-center justify-between p-3 bg-slate-900 rounded-lg border border-slate-800">
              <div>
                <span className="text-white font-bold">Hidden Layer 1 (Predator vs Prey)</span>
                <span className="text-slate-400 text-[11px] block">Linear + ReLU (Dynamic ModuleList)</span>
              </div>
              <span className="text-emerald-400 font-bold text-sm">128 Neurons</span>
            </div>

            <div className="flex items-center justify-between p-3 bg-slate-900 rounded-lg border border-slate-800">
              <div>
                <span className="text-white font-bold">Hidden Layer 2</span>
                <span className="text-slate-400 text-[11px] block">Linear + ReLU</span>
              </div>
              <span className="text-cyan-400 font-bold text-sm">64 Neurons</span>
            </div>

            <div className="flex items-center justify-between p-3 bg-slate-900 rounded-lg border border-slate-800">
              <div>
                <span className="text-white font-bold">Hidden Layer 3</span>
                <span className="text-slate-400 text-[11px] block">Linear + ReLU</span>
              </div>
              <span className="text-cyan-400 font-bold text-sm">32 Neurons</span>
            </div>

            <div className="flex items-center justify-between p-3 bg-slate-900 rounded-lg border border-slate-800">
              <div>
                <span className="text-white font-bold">Output Classifier</span>
                <span className="text-slate-400 text-[11px] block">Linear + Sigmoid Head</span>
              </div>
              <span className="text-rose-400 font-bold text-sm">1 Output (Logit)</span>
            </div>
          </div>
        </div>

        {/* Top 10 Feature Importance */}
        <div className="col-span-6 bg-[#111827] rounded-xl border border-slate-800 p-5 shadow-xl">
          <h3 className="text-sm font-bold text-white font-mono flex items-center gap-2 mb-4">
            <BarChart3 className="w-4 h-4 text-amber-400" />
            TOP INTRUSION SIGNATURE FEATURES (MUTUAL INFO)
          </h3>

          <div className="space-y-2 font-mono text-xs overflow-y-auto max-h-72 pr-1">
            {topFeatures.map((feat, i) => (
              <div
                key={feat.name}
                className="flex items-center justify-between p-2.5 bg-slate-900 rounded-lg border border-slate-800"
              >
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-slate-400 font-bold">#{i + 1}</span>
                    <span className="text-white font-bold">{feat.name}</span>
                  </div>
                  <span className="text-slate-400 text-[10px]">{feat.desc}</span>
                </div>
                <span className="text-cyan-300 font-mono font-bold">MI: {feat.mi}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
