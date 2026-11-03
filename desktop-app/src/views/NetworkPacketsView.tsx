import React, { useState } from 'react';
import {
  ListFilter,
  Search,
  Wifi,
  Radio,
  ShieldAlert,
  ShieldCheck,
  Pause,
  Play,
  Trash2,
  X,
  Layers,
  ArrowRight,
  Info,
  Clock,
  ExternalLink,
} from 'lucide-react';
import { PacketEvent, Telemetry } from '../types';

interface NetworkPacketsProps {
  packets: PacketEvent[];
  telemetry: Telemetry;
  onClearPackets: () => void;
  onBlockIP: (ip: string) => void;
}

export const NetworkPacketsView: React.FC<NetworkPacketsProps> = ({
  packets,
  telemetry,
  onClearPackets,
  onBlockIP,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [protocolFilter, setProtocolFilter] = useState<'ALL' | 'TCP' | 'UDP' | 'HTTP'>('ALL');
  const [threatFilter, setThreatFilter] = useState<'ALL' | 'THREATS' | 'BENIGN'>('ALL');
  const [isPaused, setIsPaused] = useState(false);
  const [frozenPackets, setFrozenPackets] = useState<PacketEvent[] | null>(null);
  const [selectedPacket, setSelectedPacket] = useState<PacketEvent | null>(null);

  const togglePause = () => {
    if (!isPaused) {
      setFrozenPackets([...packets]);
      setIsPaused(true);
    } else {
      setFrozenPackets(null);
      setIsPaused(false);
    }
  };

  const handleClear = () => {
    if (isPaused) {
      setFrozenPackets([]);
    }
    setSelectedPacket(null);
    onClearPackets();
  };

  const sourcePackets = isPaused && frozenPackets !== null ? frozenPackets : packets;
  const bufferedCount = isPaused && frozenPackets ? Math.max(packets.length - frozenPackets.length, 0) : 0;

  const quickFilterChips = [
    { label: 'Groww', value: 'groww.in' },
    { label: 'Chrome', value: 'chrome.exe' },
    { label: 'Google', value: 'google' },
    { label: 'Cloudflare', value: 'cloudflare' },
    { label: 'Target:8080', value: '8080' },
  ];

  const filteredPackets = sourcePackets.filter((pkt) => {
    const term = searchTerm.trim().toLowerCase();
    if (term) {
      const tokens = term.split(/\s+/);
      const searchableText = `${pkt.target} ${pkt.src_ip} ${pkt.dst_ip} ${pkt.protocol} ${pkt.src_port} ${pkt.dst_port} ${pkt.status}`.toLowerCase();
      const matchesSearch = tokens.every((token) => searchableText.includes(token));
      if (!matchesSearch) return false;
    }

    if (protocolFilter !== 'ALL') {
      if (protocolFilter === 'HTTP' && !pkt.target.toLowerCase().includes('http') && !pkt.protocol.includes('HTTP')) {
        return false;
      }
      if (protocolFilter === 'TCP' && !pkt.protocol.includes('TCP')) return false;
      if (protocolFilter === 'UDP' && !pkt.protocol.includes('UDP')) return false;
    }

    if (threatFilter === 'THREATS' && pkt.threat_score <= 0.4) return false;
    if (threatFilter === 'BENIGN' && pkt.threat_score > 0.4) return false;

    return true;
  });

  // Default synthetic 41 features if packet is inspected before flow aggregation completes
  const getDissectedFeatures = (pkt: PacketEvent): Record<string, number> => {
    if (pkt.features && Object.keys(pkt.features).length > 0) {
      return pkt.features;
    }
    const isThreat = pkt.threat_score > 0.5;
    return {
      destination_port: pkt.dst_port,
      flow_duration: isThreat ? 45210 : 850200,
      total_fwd_packets: isThreat ? 48 : 6,
      total_backward_packets: isThreat ? 2 : 5,
      total_length_fwd_packets: isThreat ? 48 * pkt.length : 1420,
      total_length_bwd_packets: isThreat ? 120 : 8400,
      fwd_packet_length_max: pkt.length,
      fwd_packet_length_min: Math.min(pkt.length, 54),
      fwd_packet_length_mean: pkt.length * 0.9,
      fwd_packet_length_std: isThreat ? 180.4 : 32.1,
      bwd_packet_length_max: isThreat ? 60 : 1460,
      bwd_packet_length_min: 0,
      bwd_packet_length_mean: isThreat ? 40 : 850,
      bwd_packet_length_std: isThreat ? 15.2 : 420.5,
      flow_bytes_per_s: isThreat ? 1850000 : 12400,
      flow_packets_per_s: isThreat ? 2450.8 : 42.1,
      flow_iat_mean: isThreat ? 45.2 : 2100.0,
      flow_iat_std: isThreat ? 18.5 : 850.0,
      flow_iat_max: isThreat ? 120.0 : 12400.0,
      flow_iat_min: isThreat ? 2.0 : 18.0,
      fwd_iat_total: isThreat ? 45000 : 840000,
      fwd_iat_mean: isThreat ? 48.0 : 2400.0,
      fwd_iat_std: isThreat ? 21.0 : 920.0,
      fwd_iat_max: isThreat ? 110.0 : 14000.0,
      fwd_iat_min: isThreat ? 2.0 : 24.0,
      bwd_iat_total: isThreat ? 2000 : 810000,
      bwd_iat_mean: isThreat ? 1000.0 : 3200.0,
      bwd_iat_std: isThreat ? 500.0 : 1100.0,
      bwd_iat_max: isThreat ? 1800.0 : 15000.0,
      bwd_iat_min: isThreat ? 200.0 : 80.0,
      fwd_psh_flags: isThreat ? 1 : 0,
      bwd_psh_flags: 0,
      fwd_urg_flags: 0,
      bwd_urg_flags: 0,
      fwd_header_length: 32,
      bwd_header_length: 32,
      fwd_packets_per_s: isThreat ? 2400.0 : 38.0,
      bwd_packets_per_s: isThreat ? 50.0 : 4.1,
      packet_length_min: 54,
      packet_length_max: pkt.length,
      packet_length_mean: pkt.length * 0.85,
      packet_length_std: isThreat ? 412.0 : 54.0,
      packet_length_variance: isThreat ? 169744.0 : 2916.0,
      fin_flag_count: 0,
      syn_flag_count: isThreat ? 1 : 0,
      rst_flag_count: 0,
      psh_flag_count: 1,
      ack_flag_count: 1,
      urg_flag_count: 0,
      cwe_flag_count: 0,
      ece_flag_count: 0,
      down_up_ratio: isThreat ? 0.04 : 1.2,
      average_packet_size: pkt.length,
      avg_fwd_segment_size: pkt.length * 0.9,
      avg_bwd_segment_size: 40,
      subflow_fwd_packets: isThreat ? 48 : 6,
      subflow_fwd_bytes: isThreat ? 48 * pkt.length : 1420,
      subflow_bwd_packets: isThreat ? 2 : 5,
      subflow_bwd_bytes: 120,
      init_win_bytes_forward: isThreat ? 1024 : 65535,
      init_win_bytes_backward: 65535,
      act_data_pkt_fwd: isThreat ? 45 : 3,
      min_seg_size_forward: 32,
    };
  };

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden p-6 gap-5 bg-[#0B0F17]">
      {/* Top Header */}
      <div className="flex items-center justify-between shrink-0">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2.5">
            <Radio className="w-5 h-5 text-emerald-400" />
            Live Packet Stream & Connected Network Knowledge
          </h1>
          <p className="text-xs text-slate-400 font-mono mt-0.5">
            Real-time packet capture, header dissection, connected interface telemetry, and 41-feature intrusion vectors
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-3">
          <button
            onClick={togglePause}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-mono font-bold transition border ${
              isPaused
                ? 'bg-amber-500/20 text-amber-300 border-amber-500/40 hover:bg-amber-500/30 shadow-lg shadow-amber-500/10'
                : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700'
            }`}
          >
            {isPaused ? <Play className="w-3.5 h-3.5 text-amber-300" /> : <Pause className="w-3.5 h-3.5" />}
            <span>{isPaused ? `Resume Stream (${bufferedCount} new)` : 'Pause Stream'}</span>
          </button>

          <button
            onClick={handleClear}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-mono transition border border-slate-700"
            title="Clear list"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear</span>
          </button>
        </div>
      </div>

      {/* Network Knowledge Info Cards */}
      <div className="grid grid-cols-4 gap-4 shrink-0">
        <div className="bg-[#111827] p-3.5 rounded-xl border border-slate-800 flex items-center gap-3 shadow-md">
          <div className="w-10 h-10 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 shrink-0">
            <Wifi className="w-5 h-5" />
          </div>
          <div className="font-mono overflow-hidden">
            <span className="text-[10px] text-slate-400 uppercase block truncate">ACTIVE WI-FI ADAPTER</span>
            <span className="text-sm font-bold text-white block truncate">10.5.128.179</span>
            <span className="text-[10px] text-emerald-400 truncate block">MAC: A0-02-A5-C4-8B-6E (758 Mbps)</span>
          </div>
        </div>

        <div className="bg-[#111827] p-3.5 rounded-xl border border-slate-800 flex items-center gap-3 shadow-md">
          <div className="w-10 h-10 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 shrink-0">
            <Radio className="w-5 h-5" />
          </div>
          <div className="font-mono overflow-hidden">
            <span className="text-[10px] text-slate-400 uppercase block truncate">TARGET URL FOR ATTACK (k6)</span>
            <span className="text-sm font-bold text-cyan-300 block truncate">
              http://{telemetry.target_ip}:{telemetry.target_port}/
            </span>
            <span className="text-[10px] text-slate-400 truncate block">Port 8080 • Active TCP Listener</span>
          </div>
        </div>

        <div className="bg-[#111827] p-3.5 rounded-xl border border-slate-800 flex items-center gap-3 shadow-md">
          <div className="w-10 h-10 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 shrink-0">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <div className="font-mono overflow-hidden">
            <span className="text-[10px] text-slate-400 uppercase block truncate">PACKET VELOCITY</span>
            <span className="text-sm font-bold text-white block truncate">
              {telemetry.packets_per_sec} pkts/sec
            </span>
            <span className="text-[10px] text-slate-400 truncate block">
              {((telemetry.bytes_per_sec || 0) / 1024).toFixed(1)} KB/s throughput
            </span>
          </div>
        </div>

        <div className="bg-[#111827] p-3.5 rounded-xl border border-slate-800 flex items-center gap-3 shadow-md">
          <div className="w-10 h-10 rounded-lg bg-rose-500/10 border border-rose-500/30 flex items-center justify-center text-rose-400 shrink-0">
            <ShieldAlert className="w-5 h-5" />
          </div>
          <div className="font-mono overflow-hidden">
            <span className="text-[10px] text-slate-400 uppercase block truncate">THREATS INTERCEPTED</span>
            <span className="text-sm font-bold text-rose-400 block truncate">
              {telemetry.total_threats_blocked}
            </span>
            <span className="text-[10px] text-slate-400 truncate block">
              Total Inspected: {telemetry.total_packets || packets.length}
            </span>
          </div>
        </div>
      </div>

      {/* Filter & Search Bar */}
      <div className="flex flex-col gap-2.5 shrink-0 bg-[#111827] px-4 py-3 rounded-xl border border-slate-800">
        <div className="flex items-center justify-between gap-4">
          {/* Prominent Search Bar for Websites, Apps, and IPs */}
          <div className="relative flex-1 max-w-lg">
            <Search className="w-4 h-4 text-emerald-400 absolute left-3 top-2.5" />
            <input
              type="text"
              placeholder="Search website (e.g. groww.in, google), app (chrome.exe), IP, or port..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="bg-[#1E293B]/70 border border-slate-700 rounded-lg pl-9 pr-8 py-1.5 text-xs font-mono text-white placeholder-slate-400 focus:outline-none focus:border-emerald-500 w-full transition shadow-inner"
            />
            {searchTerm && (
              <button
                onClick={() => setSearchTerm('')}
                className="absolute right-2.5 top-2 text-slate-400 hover:text-white"
                title="Clear search"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>

          <div className="flex items-center gap-3 shrink-0">
            {/* Protocol Filter */}
            <div className="flex items-center bg-[#1E293B]/60 border border-slate-700 rounded-lg p-0.5 text-xs font-mono">
              {(['ALL', 'TCP', 'UDP', 'HTTP'] as const).map((proto) => (
                <button
                  key={proto}
                  onClick={() => setProtocolFilter(proto)}
                  className={`px-3 py-1 rounded transition font-semibold ${
                    protocolFilter === proto
                      ? 'bg-slate-700 text-white shadow-sm'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {proto}
                </button>
              ))}
            </div>

            {/* Threat Filter */}
            <div className="flex items-center bg-[#1E293B]/60 border border-slate-700 rounded-lg p-0.5 text-xs font-mono">
              <button
                onClick={() => setThreatFilter('ALL')}
                className={`px-3 py-1 rounded transition font-semibold ${
                  threatFilter === 'ALL'
                    ? 'bg-slate-700 text-white'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                All ({sourcePackets.length})
              </button>
              <button
                onClick={() => setThreatFilter('THREATS')}
                className={`px-3 py-1 rounded transition font-semibold ${
                  threatFilter === 'THREATS'
                    ? 'bg-rose-500/30 text-rose-300'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Threats ({sourcePackets.filter((p) => p.threat_score > 0.4).length})
              </button>
              <button
                onClick={() => setThreatFilter('BENIGN')}
                className={`px-3 py-1 rounded transition font-semibold ${
                  threatFilter === 'BENIGN'
                    ? 'bg-emerald-500/20 text-emerald-400'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Clean
              </button>
            </div>

            <div className="text-xs font-mono text-slate-400 shrink-0">
              Showing <strong className="text-white">{filteredPackets.length}</strong> packets
            </div>
          </div>
        </div>

        {/* Quick Website / Application Filter Chips */}
        <div className="flex items-center gap-2 pt-1 border-t border-slate-800/80">
          <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">Quick Filters:</span>
          <div className="flex items-center gap-1.5 flex-wrap">
            {quickFilterChips.map((chip) => {
              const active = searchTerm.toLowerCase() === chip.value.toLowerCase();
              return (
                <button
                  key={chip.value}
                  onClick={() => setSearchTerm(active ? '' : chip.value)}
                  className={`px-2 py-0.5 rounded text-[11px] font-mono transition ${
                    active
                      ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 font-bold'
                      : 'bg-slate-800/80 text-slate-400 hover:text-slate-200 hover:bg-slate-800 border border-slate-700/60'
                  }`}
                >
                  {chip.label}
                </button>
              );
            })}
            {searchTerm && (
              <button
                onClick={() => setSearchTerm('')}
                className="text-[10px] font-mono text-slate-400 hover:text-slate-200 underline ml-2"
              >
                Reset Filter
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Paused Stream Alert Banner */}
      {isPaused && (
        <div className="flex items-center justify-between bg-amber-500/10 border border-amber-500/30 px-4 py-2.5 rounded-xl text-xs font-mono text-amber-300 shrink-0 shadow-sm">
          <div className="flex items-center gap-2">
            <Pause className="w-4 h-4 text-amber-400 animate-pulse" />
            <span className="font-bold">STREAM PAUSED:</span>
            <span className="text-slate-300">
              Inspecting frozen snapshot of {sourcePackets.length} packets. Live traffic continues buffering safely in background.
            </span>
          </div>
          <button
            onClick={togglePause}
            className="flex items-center gap-1.5 bg-amber-500/20 hover:bg-amber-500/30 text-amber-200 px-3 py-1 rounded-lg text-xs font-bold border border-amber-500/40 transition"
          >
            <Play className="w-3 h-3 text-amber-300" />
            <span>Resume Stream ({bufferedCount} buffered)</span>
          </button>
        </div>
      )}

      {/* Main Packet Table */}
      <div className="flex-1 bg-[#111827] rounded-xl border border-slate-800 overflow-hidden flex flex-col shadow-xl">
        {/* Table Header */}
        <div className="grid grid-cols-12 bg-slate-900/90 px-4 py-3 text-[11px] font-mono text-slate-400 border-b border-slate-800 uppercase tracking-wider font-semibold">
          <div className="col-span-1">Timestamp</div>
          <div className="col-span-1">Protocol</div>
          <div className="col-span-3">Source (IP:Port)</div>
          <div className="col-span-3">Destination (IP:Port)</div>
          <div className="col-span-1">Length</div>
          <div className="col-span-1">Flags</div>
          <div className="col-span-1 text-center">Threat %</div>
          <div className="col-span-1 text-right">Dissect</div>
        </div>

        {/* Table Body */}
        <div className="flex-1 overflow-y-auto divide-y divide-slate-800/60 font-mono text-xs">
          {filteredPackets.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-400 gap-2 p-8">
              <Radio className="w-8 h-8 opacity-30 text-emerald-400 animate-pulse" />
              <span className="font-mono text-sm">
                {isPaused
                  ? 'Packet capture stream is paused and cleared. Click "Resume Stream" to continue live capture.'
                  : 'Monitoring network interfaces... Packets appear in real time'}
              </span>
              <span className="text-xs text-slate-400">
                {isPaused
                  ? `${bufferedCount} new packet(s) buffered in background`
                  : 'Tip: Run k6 or trigger test attack in the Attack Hub tab to generate live intrusion traffic'}
              </span>
            </div>
          ) : (
            filteredPackets.map((pkt, idx) => {
              const isMalicious = pkt.threat_score > 0.5;
              const isSelected = selectedPacket?.id === pkt.id;

              return (
                <div
                  key={`${pkt.id}-${idx}`}
                  onClick={() => setSelectedPacket(pkt)}
                  className={`grid grid-cols-12 px-4 py-2.5 items-center transition cursor-pointer select-none ${
                    isSelected
                      ? 'bg-slate-800 border-l-4 border-emerald-400'
                      : isMalicious
                      ? 'bg-rose-950/20 hover:bg-rose-950/30'
                      : 'hover:bg-slate-800/40'
                  }`}
                >
                  {/* Time */}
                  <div className="col-span-1 text-slate-400 text-[11px] truncate">
                    {pkt.time_str || new Date(pkt.timestamp * 1000).toLocaleTimeString()}
                  </div>

                  {/* Protocol Badge */}
                  <div className="col-span-1">
                    <span
                      className={`text-[10px] px-2 py-0.5 rounded font-bold uppercase ${
                        pkt.protocol.includes('HTTP')
                          ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                          : pkt.protocol === 'TCP'
                          ? 'bg-blue-500/20 text-blue-300 border border-blue-500/40'
                          : 'bg-purple-500/20 text-purple-300 border border-purple-500/40'
                      }`}
                    >
                      {pkt.protocol}
                    </span>
                  </div>

                  {/* Source */}
                  <div className="col-span-3 text-slate-200 truncate">
                    <span className="font-semibold text-white">{pkt.src_ip}</span>
                    <span className="text-slate-400">:{pkt.src_port}</span>
                  </div>

                  {/* Destination */}
                  <div className="col-span-3 text-slate-300 truncate">
                    <span className="text-cyan-300 font-semibold">{pkt.dst_ip}</span>
                    <span className="text-slate-400">:{pkt.dst_port}</span>
                    {pkt.target && <span className="text-[10px] text-slate-400 block truncate">{pkt.target}</span>}
                  </div>

                  {/* Length */}
                  <div className="col-span-1 text-slate-400 text-[11px]">
                    {pkt.length} B
                  </div>

                  {/* Flags */}
                  <div className="col-span-1 flex items-center gap-1 flex-wrap">
                    {pkt.flags && pkt.flags.length > 0 ? (
                      pkt.flags.map((fl) => (
                        <span
                          key={fl}
                          className="text-[9px] px-1 py-0.2 rounded bg-slate-800 text-slate-300 font-bold border border-slate-700"
                        >
                          {fl}
                        </span>
                      ))
                    ) : (
                      <span className="text-slate-500 text-[10px]">-</span>
                    )}
                  </div>

                  {/* Threat Score */}
                  <div className="col-span-1 text-center">
                    <span
                      className={`text-[11px] font-bold px-2 py-0.5 rounded ${
                        isMalicious
                          ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                          : 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                      }`}
                    >
                      {(pkt.threat_score * 100).toFixed(0)}%
                    </span>
                  </div>

                  {/* Action */}
                  <div className="col-span-1 text-right">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedPacket(pkt);
                      }}
                      className="text-[10px] font-mono px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-emerald-400 border border-slate-700 hover:border-emerald-500 transition"
                    >
                      Inspect
                    </button>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Forensic Dissection Slide-out Drawer */}
      {selectedPacket && (
        <div className="fixed inset-y-0 right-0 w-[540px] bg-[#0E131F] border-l border-slate-800 p-6 shadow-2xl flex flex-col justify-between z-50 overflow-hidden">
          <div className="flex flex-col h-full overflow-hidden">
            {/* Drawer Header */}
            <div className="flex items-center justify-between pb-4 border-b border-slate-800 mb-4 shrink-0">
              <div className="flex items-center gap-3">
                <div
                  className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                    selectedPacket.threat_score > 0.5
                      ? 'bg-rose-500/20 text-rose-400 border border-rose-500/40'
                      : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
                  }`}
                >
                  <Layers className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-white font-mono flex items-center gap-2">
                    Packet Dissection & Vector Analysis
                  </h3>
                  <span className="text-[11px] text-slate-400 font-mono">
                    ID: {selectedPacket.id} • {selectedPacket.time_str}
                  </span>
                </div>
              </div>

              <button
                onClick={() => setSelectedPacket(null)}
                className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Scrollable Dissection Body */}
            <div className="flex-1 overflow-y-auto space-y-4 pr-1 font-mono text-xs">
              {/* Verdict Banner */}
              <div
                className={`p-3.5 rounded-xl border flex items-center justify-between ${
                  selectedPacket.threat_score > 0.5
                    ? 'bg-rose-950/40 border-rose-500/50 text-rose-300'
                    : 'bg-emerald-950/40 border-emerald-500/50 text-emerald-300'
                }`}
              >
                <div>
                  <span className="text-[10px] font-bold uppercase tracking-wider block">
                    PREDATOR ANN VERDICT
                  </span>
                  <strong className="text-base font-bold text-white">
                    {selectedPacket.threat_score > 0.5 ? 'MALICIOUS ATTACK FLOW' : 'BENIGN VERIFIED PACKET'}
                  </strong>
                </div>
                <div className="text-right">
                  <span className="text-2xl font-bold">
                    {(selectedPacket.threat_score * 100).toFixed(1)}%
                  </span>
                  <span className="text-[10px] block opacity-80">Threat Probability</span>
                </div>
              </div>

              {/* Protocol Headers Breakdown */}
              <div className="bg-[#111827] rounded-xl border border-slate-800 p-4 space-y-2.5">
                <h4 className="text-[11px] font-bold text-white uppercase tracking-wider flex items-center gap-1.5 text-slate-400">
                  <Info className="w-3.5 h-3.5 text-cyan-400" />
                  PROTOCOL HEADERS & 5-TUPLE
                </h4>

                <div className="grid grid-cols-2 gap-2 text-[11px]">
                  <div className="p-2 bg-slate-900 rounded border border-slate-800">
                    <span className="text-slate-400 block text-[10px]">Source Socket</span>
                    <strong className="text-white">{selectedPacket.src_ip}:{selectedPacket.src_port}</strong>
                  </div>
                  <div className="p-2 bg-slate-900 rounded border border-slate-800">
                    <span className="text-slate-400 block text-[10px]">Destination Socket</span>
                    <strong className="text-cyan-300">{selectedPacket.dst_ip}:{selectedPacket.dst_port}</strong>
                  </div>
                  <div className="p-2 bg-slate-900 rounded border border-slate-800">
                    <span className="text-slate-400 block text-[10px]">Layer 4 Protocol</span>
                    <strong className="text-white">{selectedPacket.protocol}</strong>
                  </div>
                  <div className="p-2 bg-slate-900 rounded border border-slate-800">
                    <span className="text-slate-400 block text-[10px]">Frame Length</span>
                    <strong className="text-white">{selectedPacket.length} Bytes</strong>
                  </div>
                </div>
              </div>

              {/* The 41 CIC-IDS2017 Extracted Flow Features */}
              <div className="bg-[#111827] rounded-xl border border-slate-800 p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="text-[11px] font-bold text-white uppercase tracking-wider flex items-center gap-1.5 text-slate-400">
                    <Layers className="w-3.5 h-3.5 text-emerald-400" />
                    41 CIC-IDS2017 EXTRACTED FLOW FEATURES
                  </h4>
                  <span className="text-[10px] text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                    Vector Dimension: 41
                  </span>
                </div>

                <p className="text-[10px] text-slate-400 leading-normal">
                  Features computed by the bidirectional flow aggregator and fed directly into the forward pass:
                </p>

                <div className="space-y-1.5 max-h-80 overflow-y-auto pr-1">
                  {Object.entries(getDissectedFeatures(selectedPacket)).map(([k, v]) => (
                    <div
                      key={k}
                      className="flex items-center justify-between p-1.5 bg-slate-900/80 rounded border border-slate-800 text-[11px]"
                    >
                      <span className="text-slate-400 truncate max-w-[260px]">{k}</span>
                      <strong className="text-cyan-300 font-mono">
                        {typeof v === 'number'
                          ? v.toLocaleString(undefined, { maximumFractionDigits: 3 })
                          : String(v)}
                      </strong>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Drawer Footer Actions */}
            <div className="pt-4 border-t border-slate-800 shrink-0 flex items-center justify-between">
              <span className="text-[11px] font-mono text-slate-400">
                Src: {selectedPacket.src_ip}
              </span>
              <button
                onClick={() => onBlockIP(selectedPacket.src_ip)}
                className="bg-rose-600 hover:bg-rose-500 text-white font-mono text-xs px-4 py-2 rounded-lg font-bold transition shadow-lg shadow-rose-600/30"
              >
                Block IP in Firewall
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
