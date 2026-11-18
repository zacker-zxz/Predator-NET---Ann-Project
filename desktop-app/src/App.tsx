import React, { useState, useEffect, useRef } from 'react';
import { Sidebar } from './components/Sidebar';
import { DefenseMonitorView } from './views/DefenseMonitorView';
import { NetworkPacketsView } from './views/NetworkPacketsView';
import { LiveFirewallView } from './views/LiveFirewallView';
import { AttackSimulatorView } from './views/AttackSimulatorView';
import { ModelStudioView } from './views/ModelStudioView';
import { AlgorithmView } from './views/AlgorithmView';
import { SettingsView } from './views/SettingsView';
import { OperationsView } from './views/OperationsView';

import {
  ActiveTab,
  FlowEvent,
  PacketEvent,
  SecurityAlert,
  Telemetry,
  LayerActivations,
} from './types';
import { fetchHistory, fetchPackets, blockIP, unblockIP, simulateAttack } from './services/api';
import {
  playTactileBeep,
  playThreatAlert,
  playBlockSuccess,
} from './services/sound';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<ActiveTab>('monitor');

  // Live Defense State
  const [telemetry, setTelemetry] = useState<Telemetry>({
    total_flows: 0,
    total_packets: 0,
    total_bytes: 0,
    total_threats_blocked: 0,
    packets_per_sec: 0,
    bytes_per_sec: 0,
    cpu_percent: 12.0,
    ram_percent: 54.0,
    gpu_name: 'NVIDIA GeForce RTX 4060 Laptop GPU',
    target_ip: '10.5.128.179',
    target_port: 8080,
    status: 'ACTIVE_DEFENSE',
    gpu_telemetry: {
      available: true,
      name: 'NVIDIA GeForce RTX 4060 Laptop GPU',
      driver_version: '617.14',
      memory_total_mb: 8188,
      memory_used_mb: 2470,
      memory_free_mb: 5718,
      temperature_c: 64,
      power_draw_w: 35.6,
      gpu_util_pct: 32,
      mem_util_pct: 18,
    },
  });

  const [flows, setFlows] = useState<FlowEvent[]>([]);
  const [packets, setPackets] = useState<PacketEvent[]>([]);
  const [activeAlerts, setActiveAlerts] = useState<SecurityAlert[]>([]);
  const [blockedIPs, setBlockedIPs] = useState<string[]>([]);

  const [threatProbability, setThreatProbability] = useState<number>(0.0);
  const [huntingMode, setHuntingMode] = useState<boolean>(false);
  const [layerActivations, setLayerActivations] = useState<LayerActivations | undefined>();
  const [predatorNeurons, setPredatorNeurons] = useState<number[]>([4, 12, 19, 27, 35, 44, 52, 61, 73, 85, 96, 108, 114, 122]);
  const [weakNeurons, setWeakNeurons] = useState<number[]>([1, 8, 15, 23, 31, 40, 48, 59, 68, 79, 89, 102, 118, 126]);

  const wsRef = useRef<WebSocket | null>(null);

  // Initialize history & packets
  useEffect(() => {
    fetchHistory().then((data) => {
      if (data) {
        if (data.recent_flows) setFlows(data.recent_flows);
        if (data.recent_packets) setPackets(data.recent_packets);
        if (data.active_alerts) setActiveAlerts(data.active_alerts);
        if (data.blocked_ips) setBlockedIPs(data.blocked_ips);
      }
    });

    fetchPackets().then((data) => {
      if (data && data.packets && data.packets.length > 0) {
        setPackets(data.packets);
      }
    });
  }, []);

  // WebSocket Connection
  useEffect(() => {
    let reconnectTimeout: any;

    const connectWS = () => {
      const wsUrl = 'ws://127.0.0.1:8765/ws/live';
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('Connected to PREDATOR-NET Live Service on port 8765');
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);

          if (msg.type === 'init_snapshot') {
            if (msg.telemetry) setTelemetry(msg.telemetry);
            if (msg.recent_flows) setFlows(msg.recent_flows);
            if (msg.recent_packets) setPackets(msg.recent_packets);
            if (msg.active_alerts) setActiveAlerts(msg.active_alerts);
            if (msg.blocked_ips) setBlockedIPs(msg.blocked_ips);
            if (msg.active_alerts) setActiveAlerts(msg.active_alerts);
          } else if (msg.type === 'block_update') {
            if (msg.blocked_ips) setBlockedIPs(msg.blocked_ips);
            if (msg.active_alerts) setActiveAlerts(msg.active_alerts);
            if (msg.telemetry) setTelemetry(msg.telemetry);
          } else if (msg.type === 'telemetry_pulse') {
            if (msg.telemetry) setTelemetry(msg.telemetry);
          } else if (msg.type === 'packet_event') {
            if (msg.packet) {
              setPackets((prev) => [msg.packet, ...prev.slice(0, 250)]);
            }
          } else if (msg.type === 'flow_event') {
            const flow: FlowEvent = msg.flow;
            if (flow) {
              setFlows((prev) => [flow, ...prev.slice(0, 150)]);
              setThreatProbability(flow.threat_probability);
              setHuntingMode(flow.hunting_mode);
              setLayerActivations(flow.layer_activations);
              if (flow.predator_neurons) setPredatorNeurons(flow.predator_neurons);
              if (flow.weak_neurons) setWeakNeurons(flow.weak_neurons);

              // Also add to packet stream
              setPackets((prev) => {
                const isBlocked = blockedIPs.includes(flow.src_ip);
                const status = isBlocked ? 'BLOCKED' : flow.is_attack ? 'THREAT_DETECTED' : 'INSPECTED';
                return [
                  {
                    id: `pkt-${Date.now()}-${prev.length}`,
                    timestamp: flow.timestamp,
                    time_str: flow.time_str,
                    protocol: flow.protocol,
                    src_ip: flow.src_ip,
                    src_port: flow.src_port,
                    dst_ip: flow.dst_ip,
                    dst_port: flow.dst_port,
                    length: flow.byte_count,
                    flags: flow.is_attack ? ['SYN', 'ACK'] : ['ACK'],
                    threat_score: flow.threat_probability,
                    status: status,
                    target: flow.target,
                    features: flow.features,
                  },
                  ...prev.slice(0, 250),
                ];
              });

              if (flow.is_attack) {
                playThreatAlert();
                setActiveAlerts((prev) => {
                  const isBlocked = blockedIPs.includes(flow.src_ip);
                  const isDup = prev.length > 0 && prev[0].src_ip === flow.src_ip && Math.abs(Date.now() - prev[0].timestamp * 1000) < 1500;
                  if (isDup) return prev;
                  return [
                    {
                      id: `alert-${Date.now()}`,
                      timestamp: flow.timestamp,
                      time_str: flow.time_str,
                      src_ip: flow.src_ip,
                      dst_ip: flow.dst_ip,
                      target: flow.target,
                      attack_type: flow.attack_type,
                      confidence: flow.confidence,
                      threat_probability: flow.threat_probability,
                      mitigated: isBlocked,
                    },
                    ...prev.slice(0, 50),
                  ];
                });
              } else {
                playTactileBeep();
              }
            }

            if (msg.telemetry) setTelemetry(msg.telemetry);
            if (msg.active_alerts) setActiveAlerts(msg.active_alerts);
            if (msg.blocked_ips) setBlockedIPs(msg.blocked_ips);
          }
        } catch (e) {
          console.error('Error parsing WS message:', e);
        }
      };

      ws.onclose = () => {
        reconnectTimeout = setTimeout(connectWS, 2000);
      };

      ws.onerror = () => {
        ws.close();
      };
    };

    connectWS();

    return () => {
      clearTimeout(reconnectTimeout);
      if (wsRef.current) wsRef.current.close();
    };
  }, [blockedIPs]);

  const handleBlockIP = async (ip: string) => {
    playBlockSuccess();
    setBlockedIPs((prev) => (prev.includes(ip) ? prev : [...prev, ip]));
    setActiveAlerts((prev) =>
      prev.map((a) => (a.src_ip === ip ? { ...a, mitigated: true } : a))
    );
    setPackets((prev) =>
      prev.map((p) => (p.src_ip === ip ? { ...p, status: 'BLOCKED' } : p))
    );
    await blockIP(ip);
  };

  const handleUnblockIP = async (ip: string) => {
    playTactileBeep();
    setBlockedIPs((prev) => prev.filter((item) => item !== ip));
    setActiveAlerts((prev) =>
      prev.map((a) => (a.src_ip === ip ? { ...a, mitigated: false } : a))
    );
    await unblockIP(ip);
  };

  const handleClearFlows = () => {
    setFlows([]);
  };

  const handleClearPackets = () => {
    setPackets([]);
  };

  const handleLaunchDemoAttack = async () => {
    await simulateAttack('ddos_flood', 50);
  };

  return (
    <div className="flex h-screen w-screen bg-[#0B0F17] text-slate-100 overflow-hidden font-sans select-none">
      {/* Sidebar Navigation */}
      <Sidebar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        telemetry={telemetry}
      />

      {/* Main View Area */}
      <main className="flex-1 flex flex-col h-full overflow-hidden bg-[#0B0F17]">
        {activeTab === 'monitor' && (
          <DefenseMonitorView
            telemetry={telemetry}
            threatProbability={threatProbability}
            huntingMode={huntingMode}
            layerActivations={layerActivations}
            predatorNeurons={predatorNeurons}
            weakNeurons={weakNeurons}
            activeAlerts={activeAlerts}
            onBlockIP={handleBlockIP}
            onLaunchDemoAttack={handleLaunchDemoAttack}
          />
        )}

        {activeTab === 'packets' && (
          <NetworkPacketsView
            packets={packets}
            telemetry={telemetry}
            onClearPackets={handleClearPackets}
            onBlockIP={handleBlockIP}
          />
        )}

        {activeTab === 'firewall' && (
          <LiveFirewallView
            flows={flows}
            blockedIPs={blockedIPs}
            onBlockIP={handleBlockIP}
            onUnblockIP={handleUnblockIP}
            onClearFlows={handleClearFlows}
          />
        )}

        {activeTab === 'attack' && (
          <AttackSimulatorView
            telemetry={telemetry}
            onAttackTriggered={() => {}}
          />
        )}

        {activeTab === 'algorithm' && <AlgorithmView />}

        {activeTab === 'studio' && <ModelStudioView />}

        {activeTab === 'operations' && <OperationsView telemetry={telemetry} />}

        {activeTab === 'settings' && <SettingsView telemetry={telemetry} />}
      </main>
    </div>
  );
};
