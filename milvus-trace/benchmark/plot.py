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
    """Plot a specific metric comparing hardware and worker configurations"""
    hardwares = ["nvme", "sata"]
    num_search_workers = [2, 10]
    
    # Create groups: (hardware, num_workers)
    groups = [(h, w) for h in hardwares for w in num_search_workers]
    group_labels = [f"{h.upper()}\n{w} workers" for h, w in groups]
    
    x = np.arange(len(groups))
    width = 0.6
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get values for each group
    values = []
    for h, w in groups:
        if w in data[h] and len(data[h][w]) > 0:
            # Average across multiple runs if they exist
            avg_value = get_average(data[h][w], metric)
            values.append(avg_value)
        else:
            values.append(0)  # Handle missing data
    
    # Create bars
    rects = ax.bar(x, values, width, label=metric)
    
    # Add value labels on bars
    for i, rect in enumerate(rects):
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=10)
    ax.set_title(f'{ylabel} Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()


def plot_results():
    """Main function to load data and create plots"""
    hardwares = ["nvme", "sata"]
    num_search_workers = [2, 10]
    
    data = {"nvme": {}, "sata": {}}
    
    for hardware in hardwares:
        for num_workers in num_search_workers:
            directory = f"milvus_{hardware}_results/{num_workers}_search_workers"
            
            # Initialize list for this configuration
            if num_workers not in data[hardware]:
                data[hardware][num_workers] = []
            
            # Check if directory exists
            if not os.path.exists(directory):
                print(f"Warning: Directory {directory} does not exist")
                continue
            
            result_files = os.listdir(directory)
            
            for filename in result_files:
                # Only process JSON files
                if not filename.endswith('.json'):
                    continue
                    
                file_path = os.path.join(directory, filename)
                
                # Check if the item is a file
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            curr_data = json.load(f)
                            data[hardware][num_workers].append(get_metrics(curr_data))
                            print(f"Loaded: {file_path}")
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")
    
    # Print summary of loaded data
    print("\n=== Data Summary ===")
    for hw in hardwares:
        for workers in num_search_workers:
            count = len(data[hw].get(workers, []))
            print(f"{hw.upper()} with {workers} workers: {count} runs")
    
    # Create plots
    plot_metric(data, "throughput", "Throughput (queries/sec)")
    plot_metric(data, "p50_lat", "P50 Latency (ms)")
    plot_metric(data, "p99_9_lat", "P99.9 Latency (ms)")
    
    plt.show()


if __name__ == "__main__":
    plot_results()