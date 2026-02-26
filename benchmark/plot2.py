import json
import matplotlib.pyplot as plt
import numpy as np
import os

def get_average(data, metric):
    """Average a metric across multiple runs"""
    ans = 0
    for run in data:
        ans += run[metric]
    return ans / len(data)

def get_metrics(data):
    """Extract metrics from result JSON"""
    results = data["results"]["search_only"]
    
    return {
        "throughput": results["actual_qps"],
        "avg_lat": results["latency"]["mean_ms"],
        "p50_lat": results["latency"]["p50_ms"],
        "p80_lat": results["latency"]["p80_ms"],
        "p90_lat": results["latency"]["p90_ms"],
        "p95_lat": results["latency"]["p95_ms"],
        "p99_lat": results["latency"]["p99_ms"],
        "p99_9_lat": results["latency"]["p99_9_ms"]
    }

def autolabel(ax, rects, fmt="{:.0f}", y_offset=-40):
    """Add value labels on bars"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90
        )

def plot_metric(data, metric, ylabel):
    x = np.arange(len(data))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 6))

    configs = list(data.keys())
    values = [data[cfg][metric] for cfg in configs]

    # Generate distinct colors
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(configs))]

    rects = []
    for i, (cfg, val) in enumerate(zip(configs, values)):
        rect = ax.bar(
            x[i],
            val,
            width,
            color=colors[i],
            label=cfg
        )
        rects.append(rect)

        # Value label
        ax.annotate(
            f"{val:.1f}",
            xy=(x[i], val),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=10)
    ax.set_title(f"{ylabel} Comparison", fontsize=14, fontweight="bold")

    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Configuration", fontsize=9)

    plt.tight_layout()



def plot_results():
    """Main function to load data and create plots"""
    file_paths = [
        "milvus_sata_results/2_search_workers/spacev1b_results_20260203_214138.json",
        "milvus_sata_results/10_search_workers/spacev1b_results_20260203_210934.json",
        "milvus_nvme_results/2_search_workers/spacev1b_results_20260204_072546.json",
        "milvus_nvme_results/10_search_workers/spacev1b_results_20260204_075411.json",
        "milvus_striped/spacev1b_results_20260211_164728.json",
        "milvus_striped_plus_256_segments/spacev1b_results_20260212_054318.json",
        "milvus_striped_plus_256_segments/spacev1b_results_20260212_055943.json"
    ]

    data = {}

    for i in range(len(file_paths)):
        fpath = file_paths[i]
        with open(fpath, 'r') as f:
            curr_data = json.load(f)
            metrics = get_metrics(curr_data)
            if i == 0:
                data["sata_2_workers"] = metrics
            elif i == 1:
                data["sata_10_workers"] = metrics
            elif i == 2:
                data["nvme_2_workers"] = metrics
            elif i == 3:
                data["nvme_10_workers"] = metrics
            elif i == 4:
                data["striped_513_segments"] = metrics
            elif i == 5:
                data["striped_256_segments_cold_cache"] = metrics
            elif i == 6:
                data["striped_256_segments_warm_cache"] = metrics       


    # Create plots
    plot_metric(data, "throughput", "Throughput (queries/sec)")
    plot_metric(data, "p50_lat", "P50 Latency (ms)")
    plot_metric(data, "p99_9_lat", "P99.9 Latency (ms)")
    
    plt.show()


if __name__ == "__main__":
    plot_results()