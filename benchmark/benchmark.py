import struct
import time
import numpy as np
from pymilvus import connections, Collection
import threading
from datetime import datetime
import json
import os
from collections import defaultdict
import statistics


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

class LatencyTracker:
    """Track detailed latency statistics"""

    def __init__(self):
        self.latencies = []

    def record(self, latency_seconds):
        self.latencies.append(latency_seconds * 1000)  # Convert to ms

    def get_stats(self):
        if not self.latencies:
            return {}

        sorted_latencies = sorted(self.latencies)
        n = len(sorted_latencies)

        return {
            "count": n,
            "mean_ms": statistics.mean(sorted_latencies),
            "p50_ms": np.percentile(sorted_latencies, 50),
            "p80_ms": np.percentile(sorted_latencies, 80),
            "p90_ms": np.percentile(sorted_latencies, 90),
            "p95_ms": np.percentile(sorted_latencies, 95),
            "p99_ms": np.percentile(sorted_latencies, 99),
            "p99_9_ms": np.percentile(sorted_latencies, 99.9),
            "min_ms": min(sorted_latencies),
            "max_ms": max(sorted_latencies)
        }

    def reset(self):
        self.latencies = []


class SearchWorker(threading.Thread):
    """Worker thread for executing search queries"""

    def __init__(self, worker_id, collection_name, queries, latency_tracker, duration, milvus_host, milvus_port):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.collection_name = collection_name
        self.queries = queries
        self.latency_tracker = latency_tracker
        self.duration = duration
        self.running = True
        self.queries_executed = 0
        self.errors = 0
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port

    def run(self):
        # Connect
        connection_alias = f"worker_{self.worker_id}"
        connections.connect(connection_alias, host=self.milvus_host, port=self.milvus_port)
        
        collection = Collection(self.collection_name, using=connection_alias)
    
        end_time = time.time() + self.duration
        query_idx = 0

        # Calculate delay between queries to achieve target QPS
        # delay = 1.0 / self.target_qps if self.target_qps > 0 else 0
        # print(f"[{time.time():.2f}] Worker {self.worker_id} starting query loop", flush=True)

        while self.running and time.time() < end_time:
            try:
                # Select a query (cycle through available queries)
                query_vector = [self.queries[query_idx % len(self.queries)].tolist()]

                # print(f"[{time.time():.2f}] Worker {self.worker_id} query {self.queries_executed}", flush=True)

                start = time.time()
                results = collection.search(
                    data=query_vector,
                    anns_field="embedding",
                    param={"metric_type": "L2", "params": {"search_list": 100}},
                    limit=TOP_K,
                    output_fields=[]
                )
                latency = time.time() - start

                self.latency_tracker.record(latency)
                self.queries_executed += 1
                query_idx += 1

                # Sleep to control QPS
                # if delay > 0:
                #    time.sleep(delay)

            except Exception as e:
                self.errors += 1
                print(f"Search worker {self.worker_id} error: {e}")

        connections.disconnect(connection_alias)

    def stop(self):
        self.running = False


class InsertWorker(threading.Thread):
    """Worker thread for inserting vectors"""

    def __init__(self, worker_id, collection_name, reader, start_idx, end_idx,
                 batch_size, target_qps, duration, milvus_host, milvus_port):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.collection_name = collection_name
        self.reader = reader
        self.current_idx = start_idx
        self.end_idx = end_idx
        self.batch_size = batch_size
        self.target_qps = target_qps
        self.duration = duration
        self.running = True
        self.vectors_inserted = 0
        self.errors = 0
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port

    def run(self):
        connection_alias = f"insert_worker_{self.worker_id}"
        connections.connect(connection_alias, host=self.milvus_host, port=self.milvus_port)
        
        collection = Collection(self.collection_name, using=connection_alias)

        end_time = time.time() + self.duration

        # Calculate delay between batches to achieve target insert rate
        # target_qps is in vectors/second, so delay = batch_size / target_qps
        delay = self.batch_size / self.target_qps if self.target_qps > 0 else 0

        while self.running and time.time() < end_time and self.current_idx < self.end_idx:
            try:
                # Calculate how many vectors to insert
                remaining = min(self.end_idx - self.current_idx,
                               self.reader.num_vectors - self.current_idx)

                if remaining <= 0:
                    # No more vectors available
                    break

                batch = min(self.batch_size, remaining)

                # Read vectors from dataset
                vectors = self.reader.read_vectors(self.current_idx, batch)

                # Check if we actually got data
                if len(vectors) == 0:
                    break

                # Prepare data for insertion
                data = [
                    {
                        "id": self.current_idx + i,
                        "vector": vectors[i].tolist()
                    }
                    for i in range(len(vectors))
                ]

                # Insert into Milvus
                self.collection.insert(collection=self.collection, data=data)

                self.vectors_inserted += len(vectors)
                self.current_idx += len(vectors)

                # Sleep to control insert rate
                if delay > 0:
                    time.sleep(delay)

            except Exception as e:
                self.errors += 1
                if self.errors < 5:  # Only print first few errors
                    print(f"Insert worker {self.worker_id} error: {e}")

        connection.disconnect(connection_alias)
    def stop(self):
        self.running = False

def run_benchmark(
    query_file,
    ground_truth_file,
    milvus_host,
    milvus_port,
    search_qps=1000,
    insert_qps=10000,
    num_search_workers=10,
    num_insert_workers=5,
    insert_batch_size=1000,
    duration=0,
    search_only_duration=300
):
    """
    Run the SPACEV1B benchmark

    Args:
        query_file: Name of query file
        ground_truth_file: Name of ground truth file
        milvus_host: Milvus server host
        milvus_port: Milvus server port
        search_qps: Target search queries per second (total across all workers)
        insert_qps: Target insert rate in vectors per second (total across all workers)
        num_search_workers: Number of concurrent search worker threads
        num_insert_workers: Number of concurrent insert worker threads
        insert_batch_size: Batch size for inserts
        duration: Duration of the concurrent search+insert phase in seconds
        search_only_duration: Duration of search-only phase (no inserts) in seconds
    """

    print("="*80)
    print("MILVUS SPACEV1B BENCHMARK")
    print("="*80)
    print(f"Configuration:")
    print(f"  Milvus: {milvus_host}:{milvus_port}")
    print(f"  Target search QPS: {search_qps:,}")
    print(f"  Target insert QPS: {insert_qps:,} vectors/sec")
    print(f"  Search workers: {num_search_workers}")
    print(f"  Insert workers: {num_insert_workers}")
    print(f"  Insert batch size: {insert_batch_size}")
    if search_only_duration > 0:
        print(f"  Search-only duration: {search_only_duration}s")
    print(f"  Concurrent phase duration: {duration}s")
    print("="*80)

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

    # Phase 3a: Search-only phase (optional)
    search_only_stats = None
    if search_only_duration > 0:
        print(f"\n{'='*80}")
        print(f"PHASE 3A: Search-Only Baseline ({search_only_duration}s)")
        print(f"{'='*80}")
        print(f"Testing search performance WITHOUT concurrent inserts")
        print()

        # Calculate QPS per worker for search-only
        # search_qps_per_worker = search_qps / num_search_workers

        # Create search-only workers
        search_only_workers = []
        for i in range(num_search_workers):
            lat_tracker = LatencyTracker()
            worker = SearchWorker(
                worker_id=i,
                collection_name=COLLECTION_NAME,
                queries=queries,
                latency_tracker=lat_tracker,
                duration=search_only_duration,
                milvus_host=milvus_host,
                milvus_port=milvus_port
            )
            search_only_workers.append(worker)

        # Start search-only workers
        start_time = time.time()
        for worker in search_only_workers:
            worker.start()

        # Monitor progress
        last_report = start_time
        while time.time() - start_time < search_only_duration:
            time.sleep(10)

            elapsed = time.time() - start_time
            total_searches = sum(w.queries_executed for w in search_only_workers)
            current_qps = total_searches / elapsed

            print(f"[{elapsed:.0f}s] Searches: {total_searches:,} ({current_qps:.0f} QPS)")

        # Stop workers
        print("\nStopping search-only workers...")
        for worker in search_only_workers:
            worker.stop()


        latencies = []
        # Wait for workers to finish
        for worker in search_only_workers:
            latencies.extend(worker.latency_tracker.latencies)
            worker.join(timeout=10)


        sorted_latencies = sorted(latencies)
        n = len(latencies)

        stats = {
            "count": n,
            "mean_ms": statistics.mean(sorted_latencies),
            "p50_ms": np.percentile(sorted_latencies, 50),
            "p80_ms": np.percentile(sorted_latencies, 80),
            "p90_ms": np.percentile(sorted_latencies, 90),
            "p95_ms": np.percentile(sorted_latencies, 95),
            "p99_ms": np.percentile(sorted_latencies, 99),
            "p99_9_ms": np.percentile(sorted_latencies, 99.9),
            "min_ms": min(sorted_latencies),
            "max_ms": max(sorted_latencies)
        }

        # Collect search-only statistics
        search_only_time = time.time() - start_time
        search_only_total = sum(w.queries_executed for w in search_only_workers)
        search_only_errors = sum(w.errors for w in search_only_workers)
        search_only_stats = {
            "duration_seconds": search_only_time,
            "total_queries": search_only_total,
            "errors": search_only_errors,
            "actual_qps": search_only_total / search_only_time,
            "latency": stats
        }

        # Print search-only results
        print("\n" + "="*80)
        print("SEARCH-ONLY PHASE RESULTS")
        print("="*80)
        print(f"\nDuration: {search_only_time:.1f}s")
        print(f"Total queries: {search_only_total:,}")
        print(f"Actual QPS: {search_only_stats['actual_qps']:.2f}")
        print(f"Errors: {search_only_errors}")
        if search_only_stats['latency']:
            lat = search_only_stats['latency']
            print(f"\nLatency Distribution:")
            print(f"  Mean:   {lat['mean_ms']:8.2f} ms")
            print(f"  P50:    {lat['p50_ms']:8.2f} ms")
            print(f"  P80:    {lat['p80_ms']:8.2f} ms")
            print(f"  P90:    {lat['p90_ms']:8.2f} ms")
            print(f"  P95:    {lat['p95_ms']:8.2f} ms")
            print(f"  P99:    {lat['p99_ms']:8.2f} ms")
            print(f"  P99.9:  {lat['p99_9_ms']:8.2f} ms")

    # Phase 3b: Concurrent search and insert
    print(f"\n{'='*80}")
    print(f"PHASE {'3B' if search_only_duration > 0 else '3'}: Concurrent Search + Insert ({duration}s)")
    print(f"{'='*80}")
    if duration > 0:
        print(f"Testing search performance WITH concurrent inserts")
        print()

        # Initialize latency tracker for concurrent phase
        latency_tracker = LatencyTracker()

        # Calculate QPS per worker
        # search_qps_per_worker = search_qps / num_search_workers
        insert_qps_per_worker = insert_qps / num_insert_workers

        # Calculate insert range for each worker
        remaining_vectors = reader.num_vectors - initial_vectors
        vectors_per_insert_worker = remaining_vectors // num_insert_workers

        # Create search workers
        search_workers = []
        for i in range(num_search_workers):
            worker = SearchWorker(
                worker_id=i,
                collection_name=COLLECTION_NAME,
                queries=queries,
                latency_tracker=latency_tracker,
                duration=duration,
                milvus_host=milvus_host,
                milvus_port=milvus_port
            )
            search_workers.append(worker)

        # Create insert workers
        insert_workers = []
        for i in range(num_insert_workers):
            start_idx = initial_vectors + i * vectors_per_insert_worker
            end_idx = start_idx + vectors_per_insert_worker
            if i == num_insert_workers - 1:  # Last worker gets any remainder
                end_idx = reader.num_vectors

            worker = InsertWorker(
                worker_id=i,
                collection_name=COLLECTION_NAME,
                reader=reader,
                start_idx=start_idx,
                end_idx=end_idx,
                batch_size=insert_batch_size,
                target_qps=insert_qps_per_worker,
                duration=duration
            )
            insert_workers.append(worker)

        # Start all workers
        start_time = time.time()
        for worker in search_workers + insert_workers:
            worker.start()

        # Monitor progress
        last_report = start_time
        while time.time() - start_time < duration:
            time.sleep(10)

            elapsed = time.time() - start_time
            total_searches = sum(w.queries_executed for w in search_workers)
            total_inserts = sum(w.vectors_inserted for w in insert_workers)
            current_qps = total_searches / elapsed
            current_insert_rate = total_inserts / elapsed

            stats = latency_tracker.get_stats()
            print(f"\n[{elapsed:.0f}s] Searches: {total_searches:,} ({current_qps:.0f} QPS), "
                  f"Inserts: {total_inserts:,} ({current_insert_rate:.0f} vec/s)")
            if stats:
                print(f"        Search latency - mean: {stats['mean_ms']:.2f}ms, "
                      f"p50: {stats['p50_ms']:.2f}ms, p95: {stats['p95_ms']:.2f}ms, "
                      f"p99: {stats['p99_ms']:.2f}ms")

        # Stop all workers
        print("\n\nStopping workers...")
        for worker in search_workers + insert_workers:
            worker.stop()

        # Wait for workers to finish
        for worker in search_workers + insert_workers:
            worker.join(timeout=10)

        # Collect final statistics
        total_time = time.time() - start_time
        total_searches = sum(w.queries_executed for w in search_workers)
        total_search_errors = sum(w.errors for w in search_workers)
        total_inserts = sum(w.vectors_inserted for w in insert_workers)
        total_insert_errors = sum(w.errors for w in insert_workers)

        search_stats = latency_tracker.get_stats()

        # Print results
        print("\n" + "="*80)
        print("CONCURRENT PHASE RESULTS (Search + Insert)")
        print("="*80)

        print(f"\nDuration: {total_time:.1f}s")

        print(f"\nSEARCH OPERATIONS:")
        print(f"  Total queries: {total_searches:,}")
        print(f"  Errors: {total_search_errors}")
        print(f"  Actual QPS: {total_searches / total_time:.2f}")
        print(f"  Target QPS: {search_qps}")
        if search_stats:
            print(f"\n  Latency Distribution:")
            print(f"    Mean:   {search_stats['mean_ms']:8.2f} ms")
            print(f"    P50:    {search_stats['p50_ms']:8.2f} ms")
            print(f"    P80:    {search_stats['p80_ms']:8.2f} ms")
            print(f"    P90:    {search_stats['p90_ms']:8.2f} ms")
            print(f"    P95:    {search_stats['p95_ms']:8.2f} ms")
            print(f"    P99:    {search_stats['p99_ms']:8.2f} ms")
            print(f"    P99.9:  {search_stats['p99_9_ms']:8.2f} ms")
            print(f"    Min:    {search_stats['min_ms']:8.2f} ms")
            print(f"    Max:    {search_stats['max_ms']:8.2f} ms")

        print(f"\nINSERT OPERATIONS:")
        print(f"  Total vectors: {total_inserts:,}")
        print(f"  Errors: {total_insert_errors}")
        print(f"  Actual rate: {total_inserts / total_time:.2f} vectors/sec")
        print(f"  Target rate: {insert_qps} vectors/sec")

    # Print comparison if search-only phase was run
    # if search_only_stats and search_stats:
    #     print("\n" + "="*80)
    #     print("COMPARISON: Search-Only vs Concurrent")
    #     print("="*80)

    #     so_lat = search_only_stats['latency']
    #     con_lat = search_stats

    #     print(f"\nThroughput:")
    #     print(f"  Search-only QPS:   {search_only_stats['actual_qps']:8.2f}")
    #     print(f"  Concurrent QPS:    {total_searches / total_time:8.2f}")
    #     degradation = (1 - (total_searches / total_time) / search_only_stats['actual_qps']) * 100
    #     print(f"  QPS Degradation:   {degradation:8.1f}%")

    #     print(f"\nLatency (milliseconds):")
    #     print(f"  Metric      Search-Only    Concurrent    Impact")
    #     print(f"  --------------------------------------------------------")
    #     for metric in ['mean_ms', 'p50_ms', 'p80_ms', 'p90_ms', 'p95_ms', 'p99_ms', 'p99_9_ms']:
    #         metric_name = metric.replace('_ms', '').upper()
    #         so_val = so_lat[metric]
    #         con_val = con_lat[metric]
    #         impact = (con_val / so_val - 1) * 100
    #         print(f"  {metric_name:8s}    {so_val:8.2f}       {con_val:8.2f}      {impact:+6.1f}%")

    #     print(f"\nKey Insights:")
    #     p50_impact = (con_lat['p50_ms'] / so_lat['p50_ms'] - 1) * 100
    #     p99_impact = (con_lat['p99_ms'] / so_lat['p99_ms'] - 1) * 100
    #     print(f"  • P50 latency increased by {p50_impact:.1f}% due to concurrent inserts")
    #     print(f"  • P99 latency increased by {p99_impact:.1f}% due to concurrent inserts")
    #     print(f"  • Search QPS decreased by {degradation:.1f}% due to concurrent inserts")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "host": milvus_host,
            "port": milvus_port,
            "collection": COLLECTION_NAME,
            "search_qps_target": search_qps,
            "insert_qps_target": insert_qps,
            "num_search_workers": num_search_workers,
            "num_insert_workers": num_insert_workers,
            "insert_batch_size": insert_batch_size,
            "search_only_duration": search_only_duration,
            "concurrent_duration": duration
        },
        "results": {
            "search_only": search_only_stats,
            # "concurrent": {
            #     "duration_seconds": total_time,
            #     "search": {
            #         "total_queries": total_searches,
            #         "errors": total_search_errors,
            #         "actual_qps": total_searches / total_time,
            #         "latency": search_stats
            #     },
            #     "insert": {
            #         "total_vectors": total_inserts,
            #         "errors": total_insert_errors,
            #         "actual_rate": total_inserts / total_time
            #     }
            # }
        }
    }

    filename = f"spacev1b_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {filename}")
    print("="*80)


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
