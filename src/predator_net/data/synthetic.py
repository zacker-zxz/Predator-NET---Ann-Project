"""
Synthetic CIC-IDS2017 Dataset Generator for PREDATOR NET.

Generates realistic network flow data matching the exact CIC-IDS2017 schema (~78 columns)
with realistic benign and attack behaviors (DoS, PortScan, DDoS, Brute Force).
Enables end-to-end testing and validation of the entire pipeline.
"""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CICIDS_COLUMNS = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    "Label",
]


def generate_synthetic_cicids2017(
    n_samples: int = 10000,
    attack_ratio: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic CIC-IDS2017-like DataFrame.

    Simulates realistic benign web/DNS/SSH flows and attack patterns
    (DoS high packet rate/small size, PortScan rapid half-open SYN flows).
    """
    rng = np.random.RandomState(seed)
    n_attacks = int(n_samples * attack_ratio)
    n_benign = n_samples - n_attacks

    data = {}

    # Destination Port: Benign typically 80, 443, 22, 53; PortScan probes various
    benign_ports = rng.choice([80, 443, 22, 53, 8080], size=n_benign, p=[0.4, 0.4, 0.05, 0.1, 0.05])
    attack_ports = rng.choice([80, 443, 21, 22, 8080, 139, 445], size=n_attacks)
    ports = np.concatenate([benign_ports, attack_ports])
    data["Destination Port"] = ports

    # Flow Duration (microseconds)
    benign_dur = rng.exponential(scale=500000, size=n_benign) + 100
    attack_dur = rng.exponential(scale=50000, size=n_attacks) + 10  # Attacks often faster
    data["Flow Duration"] = np.concatenate([benign_dur, attack_dur])

    # Packet counts
    benign_fwd_pkts = rng.poisson(lam=10, size=n_benign) + 1
    attack_fwd_pkts = rng.poisson(lam=50, size=n_attacks) + 1
    fwd_pkts = np.concatenate([benign_fwd_pkts, attack_fwd_pkts])
    data["Total Fwd Packets"] = fwd_pkts

    benign_bwd_pkts = rng.poisson(lam=12, size=n_benign)
    attack_bwd_pkts = rng.poisson(lam=2, size=n_attacks)  # Attacks often asymmetric
    bwd_pkts = np.concatenate([benign_bwd_pkts, attack_bwd_pkts])
    data["Total Backward Packets"] = bwd_pkts

    # Lengths
    data["Total Length of Fwd Packets"] = fwd_pkts * rng.uniform(40, 1400, size=n_samples)
    data["Total Length of Bwd Packets"] = bwd_pkts * rng.uniform(40, 1400, size=n_samples)

    # Packet length stats
    data["Fwd Packet Length Max"] = rng.uniform(100, 1500, size=n_samples)
    data["Fwd Packet Length Min"] = rng.uniform(0, 60, size=n_samples)
    data["Fwd Packet Length Mean"] = rng.uniform(50, 500, size=n_samples)
    data["Fwd Packet Length Std"] = rng.uniform(10, 200, size=n_samples)

    data["Bwd Packet Length Max"] = rng.uniform(100, 1500, size=n_samples)
    data["Bwd Packet Length Min"] = rng.uniform(0, 60, size=n_samples)
    data["Bwd Packet Length Mean"] = rng.uniform(50, 800, size=n_samples)
    data["Bwd Packet Length Std"] = rng.uniform(10, 300, size=n_samples)

    # Rates
    dur_sec = np.maximum(data["Flow Duration"] / 1e6, 1e-4)
    data["Flow Bytes/s"] = (data["Total Length of Fwd Packets"] + data["Total Length of Bwd Packets"]) / dur_sec
    data["Flow Packets/s"] = (fwd_pkts + bwd_pkts) / dur_sec

    # IATs
    data["Flow IAT Mean"] = data["Flow Duration"] / np.maximum(fwd_pkts + bwd_pkts - 1, 1)
    data["Flow IAT Std"] = data["Flow IAT Mean"] * 0.5
    data["Flow IAT Max"] = data["Flow IAT Mean"] * 2.0
    data["Flow IAT Min"] = data["Flow IAT Mean"] * 0.1

    data["Fwd IAT Total"] = data["Flow Duration"] * 0.9
    data["Fwd IAT Mean"] = data["Fwd IAT Total"] / np.maximum(fwd_pkts, 1)
    data["Fwd IAT Std"] = data["Fwd IAT Mean"] * 0.4
    data["Fwd IAT Max"] = data["Fwd IAT Mean"] * 1.8
    data["Fwd IAT Min"] = data["Fwd IAT Mean"] * 0.1

    data["Bwd IAT Total"] = data["Flow Duration"] * 0.8
    data["Bwd IAT Mean"] = data["Bwd IAT Total"] / np.maximum(bwd_pkts + 1, 1)
    data["Bwd IAT Std"] = data["Bwd IAT Mean"] * 0.4
    data["Bwd IAT Max"] = data["Bwd IAT Mean"] * 1.8
    data["Bwd IAT Min"] = data["Bwd IAT Mean"] * 0.1

    # Flags
    data["Fwd PSH Flags"] = rng.binomial(1, 0.05, size=n_samples)
    data["Bwd PSH Flags"] = np.zeros(n_samples)
    data["Fwd URG Flags"] = np.zeros(n_samples)
    data["Bwd URG Flags"] = np.zeros(n_samples)

    data["Fwd Header Length"] = fwd_pkts * 20
    data["Bwd Header Length"] = bwd_pkts * 20
    data["Fwd Packets/s"] = fwd_pkts / dur_sec
    data["Bwd Packets/s"] = bwd_pkts / dur_sec

    data["Min Packet Length"] = rng.uniform(0, 40, size=n_samples)
    data["Max Packet Length"] = rng.uniform(500, 1500, size=n_samples)
    data["Packet Length Mean"] = rng.uniform(60, 600, size=n_samples)
    data["Packet Length Std"] = rng.uniform(20, 300, size=n_samples)
    data["Packet Length Variance"] = data["Packet Length Std"] ** 2

    data["FIN Flag Count"] = rng.binomial(1, 0.1, size=n_samples)
    # Attacks (e.g. SYN flood / Portscan) have elevated SYN counts
    benign_syn = rng.binomial(1, 0.05, size=n_benign)
    attack_syn = rng.binomial(1, 0.65, size=n_attacks)
    data["SYN Flag Count"] = np.concatenate([benign_syn, attack_syn])

    data["RST Flag Count"] = rng.binomial(1, 0.02, size=n_samples)
    data["PSH Flag Count"] = rng.binomial(1, 0.20, size=n_samples)
    data["ACK Flag Count"] = rng.binomial(1, 0.70, size=n_samples)
    data["URG Flag Count"] = rng.binomial(1, 0.01, size=n_samples)
    data["CWE Flag Count"] = np.zeros(n_samples)
    data["ECE Flag Count"] = np.zeros(n_samples)

    data["Down/Up Ratio"] = np.round(bwd_pkts / np.maximum(fwd_pkts, 1))
    data["Average Packet Size"] = data["Packet Length Mean"]
    data["Avg Fwd Segment Size"] = data["Fwd Packet Length Mean"]
    data["Avg Bwd Segment Size"] = data["Bwd Packet Length Mean"]
    data["Fwd Header Length.1"] = data["Fwd Header Length"]

    # Bulk stats
    data["Fwd Avg Bytes/Bulk"] = np.zeros(n_samples)
    data["Fwd Avg Packets/Bulk"] = np.zeros(n_samples)
    data["Fwd Avg Bulk Rate"] = np.zeros(n_samples)
    data["Bwd Avg Bytes/Bulk"] = np.zeros(n_samples)
    data["Bwd Avg Packets/Bulk"] = np.zeros(n_samples)
    data["Bwd Avg Bulk Rate"] = np.zeros(n_samples)

    # Subflows
    data["Subflow Fwd Packets"] = fwd_pkts
    data["Subflow Fwd Bytes"] = data["Total Length of Fwd Packets"]
    data["Subflow Bwd Packets"] = bwd_pkts
    data["Subflow Bwd Bytes"] = data["Total Length of Bwd Packets"]

    data["Init_Win_bytes_forward"] = rng.choice([8192, 29200, 65535, 26883], size=n_samples)
    data["Init_Win_bytes_backward"] = rng.choice([8192, 29200, 65535, -1], size=n_samples)
    data["act_data_pkt_fwd"] = np.maximum(fwd_pkts - 1, 0)
    data["min_seg_size_forward"] = rng.choice([20, 32], size=n_samples)

    # Active / Idle
    data["Active Mean"] = rng.exponential(10000, size=n_samples)
    data["Active Std"] = data["Active Mean"] * 0.2
    data["Active Max"] = data["Active Mean"] * 1.5
    data["Active Min"] = data["Active Mean"] * 0.5

    data["Idle Mean"] = rng.exponential(50000, size=n_samples)
    data["Idle Std"] = data["Idle Mean"] * 0.2
    data["Idle Max"] = data["Idle Mean"] * 1.5
    data["Idle Min"] = data["Idle Mean"] * 0.5

    # Labels
    benign_labels = ["BENIGN"] * n_benign
    attack_types = ["DoS Hulk", "PortScan", "DDoS", "FTP-Patator", "SSH-Patator"]
    attack_labels = list(rng.choice(attack_types, size=n_attacks))
    data["Label"] = benign_labels + attack_labels

    df = pd.DataFrame(data)

    # Shuffle rows
    perm = rng.permutation(len(df))
    df = df.iloc[perm].reset_index(drop=True)

    logger.info(
        f"Generated synthetic CIC-IDS2017 dataset: {len(df)} rows, {df.shape[1]} columns. "
        f"Benign: {n_benign}, Attack: {n_attacks} ({attack_ratio*100:.1f}%)."
    )
    return df


def save_synthetic_dataset(
    output_path: Path,
    n_samples: int = 10000,
    attack_ratio: float = 0.20,
    seed: int = 42,
) -> Path:
    """Save synthetic dataset as CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_cicids2017(n_samples=n_samples, attack_ratio=attack_ratio, seed=seed)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved synthetic dataset CSV to {output_path}")
    return output_path
