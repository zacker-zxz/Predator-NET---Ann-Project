"""
PREDATOR NET — Real-time Flow Aggregator & CIC-IDS2017 Feature Extractor.

Aggregates raw TCP/UDP packets into bidirectional 5-tuple network flows
and calculates statistical features matching the 41 selected CIC-IDS2017 features.
"""

import time
import math
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


class FlowKey:
    """Canonical representation of bidirectional network flow."""

    def __init__(self, src_ip: str, src_port: int, dst_ip: str, dst_port: int, protocol: str):
        self.src_ip = src_ip
        self.src_port = src_port
        self.dst_ip = dst_ip
        self.dst_port = dst_port
        self.protocol = protocol.upper()

        # Bidirectional canonical key
        if (src_ip, src_port) <= (dst_ip, dst_port):
            self.canonical = (src_ip, src_port, dst_ip, dst_port, self.protocol)
            self.is_forward = True
        else:
            self.canonical = (dst_ip, dst_port, src_ip, src_port, self.protocol)
            self.is_forward = False

    def __hash__(self):
        return hash(self.canonical)

    def __eq__(self, other):
        return isinstance(other, FlowKey) and self.canonical == other.canonical


class FlowRecord:
    """Maintains packet history and computes flow features upon completion."""

    def __init__(self, key: FlowKey, first_packet_time: float):
        self.key = key
        self.start_time = first_packet_time
        self.last_time = first_packet_time

        # Packets
        self.fwd_packet_lengths: List[int] = []
        self.bwd_packet_lengths: List[int] = []
        self.fwd_packet_times: List[float] = []
        self.bwd_packet_times: List[float] = []
        self.packet_times: List[float] = []
        self.packet_lengths: List[int] = []

        # Flags
        self.fin_count = 0
        self.syn_count = 0
        self.rst_count = 0
        self.psh_count = 0
        self.ack_count = 0
        self.urg_count = 0

        # TCP headers / windows
        self.fwd_header_len = 0
        self.bwd_header_len = 0
        self.init_win_fwd = 0
        self.init_win_bwd = 0
        self.act_data_pkt_fwd = 0

    def add_packet(
        self,
        length: int,
        timestamp: float,
        is_forward: bool,
        flags: Optional[Dict[str, bool]] = None,
        header_len: int = 20,
        win_size: int = 0,
        payload_len: int = 0,
    ) -> None:
        """Add a single packet to this flow."""
        self.last_time = timestamp
        self.packet_times.append(timestamp)
        self.packet_lengths.append(length)

        if is_forward:
            self.fwd_packet_lengths.append(length)
            self.fwd_packet_times.append(timestamp)
            self.fwd_header_len += header_len
            if len(self.fwd_packet_lengths) == 1:
                self.init_win_fwd = win_size
            if payload_len > 0:
                self.act_data_pkt_fwd += 1
        else:
            self.bwd_packet_lengths.append(length)
            self.bwd_packet_times.append(timestamp)
            self.bwd_header_len += header_len
            if len(self.bwd_packet_lengths) == 1:
                self.init_win_bwd = win_size

        if flags:
            if flags.get("FIN"): self.fin_count += 1
            if flags.get("SYN"): self.syn_count += 1
            if flags.get("RST"): self.rst_count += 1
            if flags.get("PSH"): self.psh_count += 1
            if flags.get("ACK"): self.ack_count += 1
            if flags.get("URG"): self.urg_count += 1

    def to_features(self) -> Dict[str, float]:
        """Compute the full statistical feature dictionary."""
        duration_sec = max(self.last_time - self.start_time, 1e-6)
        duration_us = duration_sec * 1e6

        tot_fwd = len(self.fwd_packet_lengths)
        tot_bwd = len(self.bwd_packet_lengths)
        tot_pkts = tot_fwd + tot_bwd

        fwd_bytes = sum(self.fwd_packet_lengths)
        bwd_bytes = sum(self.bwd_packet_lengths)
        tot_bytes = fwd_bytes + bwd_bytes

        # Packet length stats
        def stats(arr):
            if not arr:
                return 0.0, 0.0, 0.0, 0.0, 0.0
            n = len(arr)
            mean = sum(arr) / n
            var = sum((x - mean) ** 2 for x in arr) / max(n - 1, 1)
            std = math.sqrt(var)
            return float(max(arr)), float(min(arr)), float(mean), float(std), float(var)

        f_max, f_min, f_mean, f_std, _ = stats(self.fwd_packet_lengths)
        b_max, b_min, b_mean, b_std, _ = stats(self.bwd_packet_lengths)
        p_max, p_min, p_mean, p_std, p_var = stats(self.packet_lengths)

        # IAT stats
        def iat_stats(times):
            if len(times) < 2:
                return 0.0, 0.0, 0.0, 0.0, 0.0
            iats = [(times[i] - times[i - 1]) * 1e6 for i in range(1, len(times))]
            n = len(iats)
            tot = sum(iats)
            mean = tot / n
            var = sum((x - mean) ** 2 for x in iats) / max(n - 1, 1)
            std = math.sqrt(var)
            return float(tot), float(mean), float(std), float(max(iats)), float(min(iats))

        _, flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = iat_stats(self.packet_times)
        _, _, fwd_iat_std, _, _ = iat_stats(self.fwd_packet_times)
        bwd_iat_tot, bwd_iat_mean, bwd_iat_std, bwd_iat_max, _ = iat_stats(self.bwd_packet_times)

        flow_bytes_s = tot_bytes / duration_sec
        flow_pkts_s = tot_pkts / duration_sec
        fwd_pkts_s = tot_fwd / duration_sec
        bwd_pkts_s = tot_bwd / duration_sec

        down_up_ratio = float(tot_bwd) / max(tot_fwd, 1.0)
        avg_pkt_size = tot_bytes / max(tot_pkts, 1.0)

        # Active / Idle estimations
        active_mean = min(duration_us, 1000000.0)
        active_std = 0.0
        active_max = active_mean
        active_min = active_mean
        idle_min = 0.0
        idle_std = 0.0

        return {
            "average_packet_size": avg_pkt_size,
            "total_length_fwd_packets": float(fwd_bytes),
            "fwd_packet_length_mean": f_mean,
            "packet_length_std": p_std,
            "packet_length_variance": p_var,
            "fwd_packet_length_max": f_max,
            "bwd_packet_length_mean": b_mean,
            "fwd_packet_length_std": f_std,
            "bwd_packet_length_std": b_std,
            "init_win_bytes_backward": float(self.init_win_bwd),
            "destination_port": float(self.key.dst_port),
            "flow_iat_max": flow_iat_max,
            "fwd_header_length": float(self.fwd_header_len),
            "fwd_iat_std": fwd_iat_std,
            "flow_bytes_per_s": flow_bytes_s,
            "bwd_header_length": float(self.bwd_header_len),
            "bwd_iat_max": bwd_iat_max,
            "init_win_bytes_forward": float(self.init_win_fwd),
            "flow_iat_std": flow_iat_std,
            "bwd_iat_total": bwd_iat_tot,
            "bwd_packets_per_s": bwd_pkts_s,
            "bwd_iat_mean": bwd_iat_mean,
            "act_data_pkt_fwd": float(self.act_data_pkt_fwd),
            "bwd_iat_std": bwd_iat_std,
            "flow_packets_per_s": flow_pkts_s,
            "total_fwd_packets": float(tot_fwd),
            "active_mean": active_mean,
            "active_min": active_min,
            "active_max": active_max,
            "bwd_packet_length_min": b_min,
            "min_packet_length": p_min,
            "down_up_ratio": down_up_ratio,
            "fwd_packet_length_min": f_min,
            "flow_iat_min": flow_iat_min,
            "idle_min": idle_min,
            "psh_flag_count": float(self.psh_count),
            "idle_std": idle_std,
            "ack_flag_count": float(self.ack_count),
            "active_std": active_std,
            "urg_flag_count": float(self.urg_count),
            "fwd_psh_flags": float(self.psh_count if tot_fwd > 0 else 0),
            "flow_duration": duration_us,
        }


class RealtimeFlowManager:
    """
    Tracks all active connection flows in memory and flushes flows that complete or timeout.
    """

    def __init__(self, flow_timeout_sec: float = 1.0, max_packets_per_flow: int = 100):
        self.flow_timeout_sec = flow_timeout_sec
        self.max_packets_per_flow = max_packets_per_flow
        self.active_flows: Dict[Any, FlowRecord] = {}

    def process_packet(
        self,
        src_ip: str,
        src_port: int,
        dst_ip: str,
        dst_port: int,
        protocol: str,
        length: int,
        timestamp: Optional[float] = None,
        flags: Optional[Dict[str, bool]] = None,
        header_len: int = 20,
        win_size: int = 0,
        payload_len: int = 0,
    ) -> Optional[Tuple[FlowRecord, Dict[str, float]]]:
        """
        Process a single packet. Returns a finalized flow if completed or exceeded packet limit.
        """
        now = timestamp if timestamp is not None else time.time()
        key = FlowKey(src_ip, src_port, dst_ip, dst_port, protocol)

        flow = self.active_flows.get(key.canonical)
        if flow is None:
            flow = FlowRecord(key, now)
            self.active_flows[key.canonical] = flow

        is_fwd = (src_ip, src_port) == (flow.key.src_ip, flow.key.src_port)
        flow.add_packet(
            length=length,
            timestamp=now,
            is_forward=is_fwd,
            flags=flags,
            header_len=header_len,
            win_size=win_size,
            payload_len=payload_len,
        )

        # Check for immediate flush triggers: FIN/RST or packet limit
        is_closing = flags and (flags.get("FIN") or flags.get("RST"))
        if is_closing or len(flow.packet_lengths) >= self.max_packets_per_flow:
            self.active_flows.pop(key.canonical, None)
            return flow, flow.to_features()

        return None

    def flush_expired(self, current_time: Optional[float] = None) -> List[Tuple[FlowRecord, Dict[str, float]]]:
        """Flush flows that have been idle longer than flow_timeout_sec."""
        now = current_time if current_time is not None else time.time()
        expired = []
        keys_to_remove = []

        for c_key, flow in self.active_flows.items():
            if now - flow.last_time >= self.flow_timeout_sec:
                keys_to_remove.append(c_key)
                expired.append((flow, flow.to_features()))

        for k in keys_to_remove:
            self.active_flows.pop(k, None)

        return expired
