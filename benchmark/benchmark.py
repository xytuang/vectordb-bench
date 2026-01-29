import struct
import time
import numpy as np
from pymilvus import connections, Collection
from tqdm import tqdm

COLLECTION_NAME = "spacev1b"
TOP_K = 100

def read_spacev1b_queries(query_file):
    """Read SPACEV1B query vectors"""
    with open(query_file, 'rb') as f:
        q_count = struct.unpack('i', f.read(4))[0]
        q_dimension = struct.unpack('i', f.read(4))[0]
        queries = np.frombuffer(f.read(q_count * q_dimension), dtype=np.int8).reshape((q_count, q_dimension))
        queries = queries.astype(np.float32)
    return queries

def read_spacev1b_groundtruth(truth_file):
    """Read SPACEV1B ground truth"""
    with open(truth_file, 'rb') as f:
        t_count = struct.unpack('i', f.read(4))[0]
        topk = struct.unpack('i', f.read(4))[0]
        truth_vids = np.frombuffer(f.read(t_count * topk * 4), dtype=np.int32).reshape((t_count, topk))
        truth_distances = np.frombuffer(f.read(t_count * topk * 4), dtype=np.float32).reshape((t_count, topk))
    return truth_vids, truth_distances

def calculate_recall(results, ground_truth, k=100):
    """Calculate recall@K"""
    recalls = []
    for result, truth in zip(results, ground_truth):
        result_ids = set([hit.id for hit in result])
        truth_ids = set(truth[:k])
        recall = len(result_ids & truth_ids) / min(k, len(truth_ids))
        recalls.append(recall)
    return np.mean(recalls)

def run_benchmark(query_file, ground_truth_file, milvus_host, milvus_port):
    """Run benchmark"""
    # Connect
    print(f"Connecting to Milvus at {milvus_host}:{milvus_port}")
    connections.connect("default", host=milvus_host, port=milvus_port)
    
    collection = Collection(COLLECTION_NAME)
    collection.load()
    
    # Load queries
    print(f"Loading queries from {query_file}")
    queries = read_spacev1b_queries(query_file)
    print(f"Loaded {len(queries)} queries")
    
    # Load ground truth
    print(f"Loading ground truth from {ground_truth_file}")
    ground_truth_ids, ground_truth_distances = read_spacev1b_groundtruth(ground_truth_file)
    print(f"Ground truth shape: {ground_truth_ids.shape}")
    
    # Run queries
    print("Running queries...")
    search_params = {"metric_type": "L2", "param": {"search_list": 100}}
    
    latencies = []
    all_results = []
    
    for query in tqdm(queries, desc="Querying"):
        start = time.time()
        results = collection.search(
            data=[query.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=TOP_K
        )
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)
        all_results.append(results[0])
    
    # Calculate metrics
    recall = calculate_recall(all_results, ground_truth_ids, k=TOP_K)
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    qps = len(queries) / (sum(latencies) / 1000)
    
    connections.disconnect("default")
    
    return {
        "total_queries": len(queries),
        "avg_latency": avg_latency,
        "p95_latency": p95_latency,
        "p99_latency": p99_latency,
        "qps": qps,
        "recall": recall
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python benchmark.py <query_file> <ground_truth_file>")
        sys.exit(1)
    
    results = run_benchmark(
        query_file=sys.argv[1],
        ground_truth_file=sys.argv[2],
        milvus_host="node1",
        milvus_port="19530"
    )
    print(results)
