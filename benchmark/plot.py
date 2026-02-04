import json
import matplotlib.pyplot as plt
import numpy as np

def get_average(data, metric):
    ans = 0
    for run in data:
        ans += run[metric]

    return ans / len(data)

def get_metrics(data):
    results = data["results"]["search_only"]

    return {
        "throughput": results["actual_qps"],
        
    }
def autolabel(ax, rects, fmt="{:.0f}", y_offset=-40):
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
    hardwares = ["nvme", "sata"]
    num_search_workers = [2, 10]

    x = np.arange(len(hardwares))
    width = 0.18

    offsets = (np.arange(len(num_search_workers)) - (len(num_search_workers) - 1)/2) * width
    groups = [(h) for h in hardwares]

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, scheduler in enumerate(schedulers):
        values = [data[h][w][scheduler][metric] for (h,w) in groups]
        rects = ax.bar(x + offsets[i], values, width, label=scheduler)

    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(["Optane\n99.5/0.5", "Optane\n80/20", "SATA\n99.5/0.5", "SATA\n80/20"])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()


def stat_analysis(data, hardware, workload, metric):
    print(f"============Metric: {metric}=================")
    workload_results = data[hardware][workload]
    baseline = workload_results["default"][metric]
    for scheduler, metrics in workload_results.items():
        val = metrics[metric]
        percentage_diff = ((val - baseline)/baseline) * 100
        print(f"scheduler: {scheduler}, {metric}: {val}, percentage_diff: {percentage_diff}")


def plot_results():
    hardwares = ["nvme", "sata"]
    num_search_workers = [2, 10]
    metrics = ["read_throughput", "scan_throughput", "read_avg_latency", "read_p99_latency", "scan_avg_latency", "scan_p99_latency"]

    data = {"nvme": {}, "sata": {}}

    for hardware in hardwares:
        for num_workers in num_search_workers:
            directory = f"milvus_{hardware}_results/{num_workers}_search_workers"

            result_files = os.listdir(directory)

            for filename in result_files:
                # Construct the full file path
                file_path = os.path.join(directory, filename)

                # Check if the item is a file
                if os.path.isfile(file_path):
                    # Open and process the file
                    try:
                        with open(file_path, 'r') as f:
                            curr_data = json.load(f)
                    except Exception as e:
                        print(f"An error occurred while reading {filename}: {e}")

        if num_workers not in data[hardware]:
            data[hardware][num_workers] = []

        for metric in metrics:
            data[hardware][num_workers].append(curr_data["results"]["search_only"])

    for metric in metrics:
        stat_analysis(data, "sata", "80-20", metric)
        print("\n\n\n\n")

    plot_metric(data, "read_throughput", "Average GET throughput (ops/sec)")
    plot_metric(data, "read_avg_latency", "Average GET latency (ns)")
    plot_metric(data, "read_p99_latency", "Average GET p99 latency (ns)")

    plot_metric(data, "scan_throughput", "Average SCAN throughput (ops/sec)")
    plot_metric(data, "scan_avg_latency", "Average SCAN latency (ns)")
    plot_metric(data, "scan_p99_latency", "Average SCAN p99 latency (ns)")

    plt.show()


if __name__ == "__main__":
    plot_results()
