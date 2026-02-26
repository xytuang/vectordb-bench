import json
import matplotlib.pyplot as plt
import numpy as np
import os


def get_average(runs, metric):
    """Average a metric across multiple runs"""
    return sum(run[metric] for run in runs) / len(runs)


def extract_fio_summary(data):
    """
    Extract performance-critical metrics from fio JSON output.
    """

    job = data["jobs"][0]
    read = job["read"]
    disk = data["disk_util"][0]

    return {
        "throughput_MBps": read["bw_bytes"] / (1024 ** 2),
        "iops": read["iops"],
        "latency_mean_us": read["clat_ns"]["mean"] / 1000,
        "latency_p99_us": read["clat_ns"]["percentile"]["99.000000"] / 1000,
        "latency_p999_us": read["clat_ns"]["percentile"]["99.900000"] / 1000,
        "disk_util_percent": disk["util"],
    }


def aggregate_config_runs(config_runs):
    """
    Average all runs within a configuration directory.
    """
    metrics = config_runs[0].keys()
    aggregated = {}

    for metric in metrics:
        aggregated[metric] = get_average(config_runs, metric)

    return aggregated


def plot_metric(data, metric, ylabel):
    configs = list(data.keys())
    values = [data[cfg][metric] for cfg in configs]

    x = np.arange(len(configs))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, values)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right")
    ax.set_title(f"{ylabel} Comparison")

    # Add labels on top
    for i, val in enumerate(values):
        ax.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()


def plot_results():
    """
    Load directory structure:
        ./config1/run1.json
        ./config1/run2.json
        ./config2/run1.json
        ...

    Each top-level directory = one configuration.
    """

    data = {}

    for item_name in os.listdir("."):
        full_path = os.path.join(".", item_name)

        # Only process directories
        if not os.path.isdir(full_path):
            continue

        config_runs = []

        for result_file in os.listdir(full_path):
            result_path = os.path.join(full_path, result_file)

            if not result_file.endswith(".json"):
                continue

            with open(result_path, "r") as f:
                fio_json = json.load(f)
                config_runs.append(extract_fio_summary(fio_json))

        if config_runs:
            data[item_name] = aggregate_config_runs(config_runs)

    # ---- PLOTS ----

    plot_metric(data, "throughput_MBps", "Throughput (MB/s)")
    plot_metric(data, "iops", "IOPS")
    plot_metric(data, "latency_mean_us", "Mean Latency (µs)")
    plot_metric(data, "latency_p999_us", "P99.9 Latency (µs)")
    plot_metric(data, "disk_util_percent", "Disk Utilization (%)")

    plt.show()


if __name__ == "__main__":
    plot_results()
