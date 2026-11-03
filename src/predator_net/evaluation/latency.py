"""
Inference Latency and Throughput Benchmarking for PREDATOR NET.

Per Spec §43:
  Identical hardware, batch size, and preprocessing for all models.
  Reports:
    - Single-sample latency (mean, median, p95, std)
    - Batch inference latency
    - Throughput (samples per second)
"""

import time
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn


def measure_inference_latency(
    model: nn.Module,
    input_dim: int,
    batch_size: int = 1024,
    device: str = "cpu",
    warmup_runs: int = 10,
    benchmark_runs: int = 50,
) -> Dict[str, Any]:
    """
    Measure single-sample and batched inference latency and throughput.

    Args:
        model: PyTorch model in evaluation mode.
        input_dim: Number of input features.
        batch_size: Batch size for batched throughput benchmark.
        device: "cpu" or "cuda".
        warmup_runs: Number of forward passes to warm up cache/GPU.
        benchmark_runs: Number of timed runs to aggregate.

    Returns:
        Dictionary of latency statistics in milliseconds and samples/sec throughput.
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    # 1. Single sample benchmark
    single_x = torch.randn(1, input_dim, dtype=torch.float32, device=dev)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(single_x)
            if dev.type == "cuda":
                torch.cuda.synchronize()

    single_times = []
    with torch.no_grad():
        for _ in range(benchmark_runs):
            start = time.perf_counter()
            _ = model(single_x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            single_times.append((time.perf_counter() - start) * 1000.0)  # ms

    single_arr = np.array(single_times)

    # 2. Batched benchmark
    batch_x = torch.randn(batch_size, input_dim, dtype=torch.float32, device=dev)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(batch_x)
            if dev.type == "cuda":
                torch.cuda.synchronize()

    batch_times = []
    with torch.no_grad():
        for _ in range(benchmark_runs):
            start = time.perf_counter()
            _ = model(batch_x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            batch_times.append((time.perf_counter() - start) * 1000.0)  # ms

    batch_arr = np.array(batch_times)
    avg_batch_time_sec = np.mean(batch_arr) / 1000.0
    throughput = float(batch_size / max(avg_batch_time_sec, 1e-9))

    return {
        "device": str(device),
        "single_sample_latency_ms": {
            "mean": float(np.mean(single_arr)),
            "median": float(np.median(single_arr)),
            "p95": float(np.percentile(single_arr, 95)),
            "std": float(np.std(single_arr)),
        },
        "batch_latency_ms": {
            "batch_size": batch_size,
            "mean": float(np.mean(batch_arr)),
            "median": float(np.median(batch_arr)),
        },
        "throughput_samples_per_sec": float(throughput),
    }
