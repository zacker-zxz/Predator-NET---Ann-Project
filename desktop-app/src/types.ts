export interface LayerActivations {
  input: number[];
  hidden1: number[];
  hidden2: number[];
  hidden3: number[];
  output: number;
}

export interface FlowEvent {
  timestamp: number;
  time_str: string;
  src_ip: string;
  src_port: number;
  dst_ip: string;
  dst_port: number;
  target: string;
  protocol: string;
  packet_count: number;
  byte_count: number;
  threat_probability: number;
  is_attack: boolean;
  attack_type: string;
  confidence: number;
  hunting_mode: boolean;
  layer_activations?: LayerActivations;
  predator_neurons?: number[];
  weak_neurons?: number[];
  features?: Record<string, number>;
}

export interface PacketEvent {
  id: string;
  timestamp: number;
  time_str: string;
  protocol: string;
  src_ip: string;
  src_port: number;
  dst_ip: string;
  dst_port: number;
  length: number;
  flags: string[];
  threat_score: number;
  status: 'INSPECTED' | 'BLOCKED' | 'FLAGGED';
  target: string;
  features?: Record<string, number>;
}

export interface SecurityAlert {
  id: string;
  timestamp: number;
  time_str: string;
  src_ip: string;
  dst_ip: string;
  target: string;
  attack_type: string;
  confidence: number;
  threat_probability: number;
  mitigated: boolean;
}

export interface GpuTelemetry {
  available: boolean;
  name: string;
  driver_version: string;
  memory_total_mb: number;
  memory_used_mb: number;
  memory_free_mb: number;
  temperature_c: number;
  power_draw_w: number;
  gpu_util_pct: number;
  mem_util_pct: number;
}

export interface NetworkAdapter {
  name: string;
  ip: string;
  netmask: string;
  mac: string;
  is_up: boolean;
  speed_mbps: number;
  mtu: number;
  bytes_sent: number;
  bytes_recv: number;
  packets_sent: number;
  packets_recv: number;
  is_primary: boolean;
}

export interface SystemTelemetry {
  status: string;
  timestamp: number;
  gpu: GpuTelemetry;
  adapters: NetworkAdapter[];
  host: {
    os: string;
    cpu_percent: number;
    cpu_count: number;
    ram_percent: number;
    ram_total_gb: number;
    ram_used_gb: number;
  };
  model: {
    name: string;
    topology: string;
    input_features: number;
    baseline_accuracy: string;
    baseline_precision: string;
    baseline_recall: string;
    false_positive_rate: string;
    throughput_mpps: string;
    predator_hunting_alpha: number;
    weights_status: string;
  };
  target: {
    primary_ip: string;
    port: number;
    url: string;
  };
}

export interface Telemetry {
  total_flows: number;
  total_packets: number;
  total_bytes: number;
  total_threats_blocked: number;
  packets_per_sec: number;
  bytes_per_sec: number;
  cpu_percent: number;
  ram_percent: number;
  gpu_name: string;
  target_ip: string;
  target_port: number;
  status: string;
  download_bytes_per_sec?: number;
  upload_bytes_per_sec?: number;
  gpu_telemetry?: GpuTelemetry;
}


export type ActiveTab = 'monitor' | 'packets' | 'firewall' | 'attack' | 'algorithm' | 'studio' | 'settings' | 'operations';
