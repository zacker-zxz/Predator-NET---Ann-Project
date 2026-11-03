import React, { useState } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  Search,
  Filter,
  Ban,
  Radio,
  Globe,
  Trash2,
} from 'lucide-react';
import { FlowEvent } from '../types';

interface LiveFirewallProps {
  flows: FlowEvent[];
  blockedIPs: string[];
  onBlockIP: (ip: string) => void;
  onUnblockIP?: (ip: string) => void;
  onClearFlows: () => void;
}

export const LiveFirewallView: React.FC<LiveFirewallProps> = ({
  flows,
  blockedIPs,
  onBlockIP,
  onUnblockIP,
  onClearFlows,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState<'all' | 'attacks' | 'benign'>('all');

  const filteredFlows = flows.filter((f) => {
    const matchesSearch =
      f.src_ip.includes(searchTerm) ||
      f.dst_ip.includes(searchTerm) ||
      f.target.toLowerCase().includes(searchTerm.toLowerCase()) ||
      f.attack_type.toLowerCase().includes(searchTerm.toLowerCase());

    if (!matchesSearch) return false;
    if (filterType === 'attacks') return f.is_attack;
    if (filterType === 'benign') return !f.is_attack;
    return true;
  });

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden p-6 gap-5 bg-[#0B0F17]">
      {/* Header Controls */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2.5">
            <Radio className="w-5 h-5 text-emerald-400" />
            Live Network Protector & Firewall Traffic
          </h1>
          <p className="text-xs text-slate-400 font-mono mt-0.5">
            Sniffing live packets, web traffic, and port connections evaluated by PREDATOR ANN
          </p>
        </div>

        {/* Filter & Search Bar */}
        <div className="flex items-center gap-3">
          <div className="relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
            <input
              type="text"
              placeholder="Search IP, port, or site..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="bg-[#1E293B]/60 border border-slate-700 rounded-lg pl-9 pr-4 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-emerald-500 w-64"
            />
          </div>

          <div className="flex items-center bg-[#1E293B]/60 border border-slate-700 rounded-lg p-0.5 text-xs font-mono">
            <button
              onClick={() => setFilterType('all')}
              className={`px-3 py-1 rounded transition font-semibold ${
                filterType === 'all'
                  ? 'bg-slate-700 text-white'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              All ({flows.length})
            </button>
            <button
              onClick={() => setFilterType('attacks')}
              className={`px-3 py-1 rounded transition font-semibold ${
                filterType === 'attacks'
                  ? 'bg-rose-500/30 text-rose-300'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Attacks ({flows.filter((f) => f.is_attack).length})
            </button>
            <button
              onClick={() => setFilterType('benign')}
              className={`px-3 py-1 rounded transition font-semibold ${
                filterType === 'benign'
                  ? 'bg-emerald-500/20 text-emerald-400'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Benign
            </button>
          </div>

          <button
            onClick={onClearFlows}
            className="p-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-slate-400 hover:text-rose-400 transition"
            title="Clear Table"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Main Connection Table */}
      <div className="flex-1 bg-[#111827] rounded-xl border border-slate-800 overflow-hidden flex flex-col shadow-xl">
        {/* Table Header */}
        <div className="grid grid-cols-12 bg-slate-900/90 px-4 py-3 text-[11px] font-mono text-slate-400 border-b border-slate-800 uppercase tracking-wider font-semibold">
          <div className="col-span-1">Time</div>
          <div className="col-span-1">Proto</div>
          <div className="col-span-3">Source IP & Port</div>
          <div className="col-span-3">Target Domain / Service</div>
          <div className="col-span-1">Packets</div>
          <div className="col-span-2">ANN Verdict</div>
          <div className="col-span-1 text-right">Action</div>
        </div>

        {/* Table Body */}
        <div className="flex-1 overflow-y-auto divide-y divide-slate-800/60 font-mono text-xs">
          {filteredFlows.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-400 gap-2 p-8">
              <Globe className="w-8 h-8 opacity-30 text-emerald-400" />
              <span>Awaiting incoming network flows or attack packets...</span>
            </div>
          ) : (
            filteredFlows.map((flow, idx) => {
              const isBlocked = blockedIPs.includes(flow.src_ip);
              return (
                <div
                  key={`${flow.timestamp}-${idx}`}
                  className={`grid grid-cols-12 px-4 py-2.5 items-center transition hover:bg-slate-800/40 ${
                    flow.is_attack ? 'bg-rose-950/20' : ''
                  }`}
                >
                  <div className="col-span-1 text-slate-400 text-[11px] truncate">
                    {flow.time_str}
                  </div>

                  <div className="col-span-1">
                    <span className="text-[10px] bg-slate-800 text-slate-300 border border-slate-700 px-2 py-0.5 rounded font-bold uppercase">
                      {flow.protocol}
                    </span>
                  </div>

                  <div className="col-span-3 text-white truncate">
                    <span className="font-semibold">{flow.src_ip}</span>
                    <span className="text-slate-400">:{flow.src_port}</span>
                  </div>

                  <div className="col-span-3 text-cyan-300 truncate">
                    <span className="text-slate-200">{flow.target}</span>
                  </div>

                  <div className="col-span-1 text-slate-400 text-[11px]">
                    {flow.packet_count} ({((flow.byte_count || 0) / 1024).toFixed(1)} KB)
                  </div>

                  <div className="col-span-2 flex items-center gap-2">
                    {flow.is_attack ? (
                      <span className="text-rose-400 font-bold flex items-center gap-1.5 text-[11px]">
                        <ShieldAlert className="w-3.5 h-3.5" />
                        {flow.attack_type} ({(flow.confidence * 100).toFixed(0)}%)
                      </span>
                    ) : (
                      <span className="text-emerald-400 flex items-center gap-1.5 text-[11px]">
                        <ShieldCheck className="w-3.5 h-3.5" />
                        Benign Flow
                      </span>
                    )}
                  </div>

                  <div className="col-span-1 text-right">
                    {isBlocked ? (
                      <button
                        onClick={() => onUnblockIP && onUnblockIP(flow.src_ip)}
                        className="text-[10px] text-rose-400 hover:text-emerald-300 font-bold px-2 py-1 bg-rose-500/10 hover:bg-emerald-500/10 rounded border border-rose-500/20 hover:border-emerald-500/30 transition cursor-pointer"
                        title="Click to Unblock IP from firewall"
                      >
                        BLOCKED ✕
                      </button>
                    ) : flow.is_attack ? (
                      <button
                        onClick={() => onBlockIP(flow.src_ip)}
                        className="text-[10px] bg-rose-600 hover:bg-rose-500 text-white font-bold px-2.5 py-1 rounded transition cursor-pointer"
                      >
                        Block IP
                      </button>
                    ) : (
                      <button
                        onClick={() => onBlockIP(flow.src_ip)}
                        className="text-[10px] bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white px-2 py-1 rounded transition border border-slate-700 cursor-pointer"
                        title="Add to firewall blacklist"
                      >
                        Blacklist
                      </button>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
};
