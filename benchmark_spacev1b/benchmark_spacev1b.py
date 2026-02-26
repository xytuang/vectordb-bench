#!/usr/bin/env python3
"""
Milvus Benchmark using SPACEV1B Dataset

This script benchmarks Milvus performance using the SPACEV1B dataset:
1. Load first 50M vectors into the database
2. Trigger compaction and wait for completion
3. Run concurrent search (using query set) and insert (remaining 50M vectors)
4. Measure detailed search latency distribution
"""

import time
import threading
import numpy as np
from pymilvus import MilvusClient, DataType, connections, utility
from datetime import datetime
import json
import struct
import os
from collections import defaultdict
import statistics


class SPACEV1BReader:
    """Reader for SPACEV1B dataset files"""

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.vectors_file = os.path.join(dataset_path, "vectors.bin/vectors_merged.bin")
        self.query_file = os.path.join(dataset_path, "query.bin")

        # Read dataset metadata
        self.num_vectors, self.dim = self._read_header(self.vectors_file)
        self.num_queries, self.query_dim = self._read_header(self.query_file)

        print(f"Dataset info:")
        print(f"  Vectors: {self.num_vectors:,} vectors of dimension {self.dim}")
        print(f"  Queries: {self.num_queries:,} queries of dimension {self.query_dim}")

        if self.dim != self.query_dim:
            raise ValueError(f"Dimension mismatch: vectors={self.dim}, queries={self.query_dim}")

    def _read_header(self, filepath):
        """Read the header of a binary file to get count and dimension"""
        with open(filepath, 'rb') as f:
            count = struct.unpack('i', f.read(4))[0]
            dim = struct.unpack('i', f.read(4))[0]
        return count, dim

    def read_vectors(self, start_idx, count):
        """Read a range of vectors from the merged file"""
        with open(self.vectors_file, 'rb') as f:
            # Skip header (8 bytes) and previous vectors
            header_size = 8
            bytes_per_vector = self.dim
            offset = header_size + start_idx * bytes_per_vector
            f.seek(offset)

            # Read the requested vectors
            bytes_to_read = count * bytes_per_vector
            data = f.read(bytes_to_read)

            # Convert to numpy array (int8 -> float32 for Milvus)
            vectors = np.frombuffer(data, dtype=np.int8).reshape(count, self.dim)
            return vectors.astype(np.float32)

    def read_queries(self):
        """Read all query vectors"""
        with open(self.query_file, 'rb') as f:
            # Skip header
            f.read(8)
            # Read all queries
            data = f.read()
            queries = np.frombuffer(data, dtype=np.int8).reshape(self.num_queries, self.query_dim)
            return queries.astype(np.float32)


class LatencyTracker:
    """Track detailed latency statistics"""

    def __init__(self):
        self.latencies = []
        self.lock = threading.Lock()

    def record(self, latency_seconds):
        with self.lock:
            self.latencies.append(latency_seconds * 1000)  # Convert to ms

    def get_stats(self):
        with self.lock:
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
        with self.lock:
            self.latencies = []


class SearchWorker(threading.Thread):
    """Worker thread for executing search queries"""

    def __init__(self, worker_id, client, collection_name, queries, target_qps, latency_tracker, duration):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.client = client
        self.collection_name = collection_name
        self.queries = queries
        self.target_qps = target_qps
        self.latency_tracker = latency_tracker
        self.duration = duration
        self.running = True
        self.queries_executed = 0
        self.errors = 0

    def run(self):
        end_time = time.time() + self.duration
        query_idx = 0

        # Calculate delay between queries to achieve target QPS
        delay = 1.0 / self.target_qps if self.target_qps > 0 else 0

        while self.running and time.time() < end_time:
            try:
                # Select a query (cycle through available queries)
                query_vector = [self.queries[query_idx % len(self.queries)].tolist()]

                start = time.time()
                results = self.client.search(
                    collection_name=self.collection_name,
                    data=query_vector,
                    anns_field="vector",
                    search_params={"metric_type": "L2", "params": {"nprobe": 10}},
                    limit=10,
                    output_fields=[]
                )
                latency = time.time() - start

                self.latency_tracker.record(latency)
                self.queries_executed += 1
                query_idx += 1

                # Sleep to control QPS
                if delay > 0:
                    time.sleep(delay)

            except Exception as e:
                self.errors += 1
                print(f"Search worker {self.worker_id} error: {e}")

    def stop(self):
        self.running = False


class InsertWorker(threading.Thread):
    """Worker thread for inserting vectors"""

    def __init__(self, worker_id, client, collection_name, reader, start_idx, end_idx,
                 batch_size, target_qps, duration):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.client = client
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

    def run(self):
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
                self.client.insert(collection_name=self.collection_name, data=data)

                self.vectors_inserted += len(vectors)
                self.current_idx += len(vectors)

                # Sleep to control insert rate
                if delay > 0:
                    time.sleep(delay)

            except Exception as e:
                self.errors += 1
                if self.errors < 5:  # Only print first few errors
                    print(f"Insert worker {self.worker_id} error: {e}")

    def stop(self):
        self.running = False


def load_initial_vectors(client, collection_name, reader, num_vectors, batch_size=10000):
    """Load initial vectors into the collection"""
    print(f"\nLoading {num_vectors:,} initial vectors...")

    num_batches = (num_vectors + batch_size - 1) // batch_size
    start_time = time.time()

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        count = min(batch_size, num_vectors - start_idx)

        # Read vectors
        vectors = reader.read_vectors(start_idx, count)

        # Prepare data
        data = [
            {
                "id": start_idx + i,
                "vector": vectors[i].tolist()
            }
            for i in range(count)
        ]

        # Insert
        client.insert(collection_name=collection_name, data=data)

        # Progress update
        if (batch_idx + 1) % 100 == 0 or batch_idx == num_batches - 1:
            elapsed = time.time() - start_time
            vectors_loaded = min((batch_idx + 1) * batch_size, num_vectors)
            rate = vectors_loaded / elapsed
            print(f"  Loaded {vectors_loaded:,} / {num_vectors:,} vectors "
                  f"({vectors_loaded*100.0/num_vectors:.1f}%) - "
                  f"{rate:,.0f} vectors/sec")

    total_time = time.time() - start_time
    print(f"Initial load complete in {total_time:.1f}s ({num_vectors/total_time:,.0f} vectors/sec)")


def flush_and_index(client, collection_name, host="localhost", port=19530):
    """Flush data and wait for indexing to complete"""
    print("\nFlushing data and waiting for indexing...")

    # Connect using connections API
    connections.connect("default", host=host, port=port)

    try:
        # Flush all data
        print("  Flushing data...")
        start_time = time.time()
        utility.flush_all()
        elapsed = time.time() - start_time
        print(f"  Data flushed in {elapsed:.1f}s")

        # Wait for index building to complete
        print("  Waiting for index building...")
        start_time = time.time()
        utility.wait_for_index_building_complete(collection_name)
        elapsed = time.time() - start_time
        print(f"  Index building completed in {elapsed:.1f}s")

    finally:
        connections.disconnect("default")


def run_benchmark(
    dataset_path,
    host="localhost",
    port=19530,
    collection_name="spacev1b_benchmark",
    initial_vectors=50_000_000,
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
        dataset_path: Path to SPACEV1B dataset directory
        host: Milvus server host
        port: Milvus server port
        collection_name: Name of the collection to use
        initial_vectors: Number of vectors to load initially (default 50M)
        search_qps: Target search queries per second (total across all workers)
        insert_qps: Target insert rate in vectors per second (total across all workers)
        num_search_workers: Number of concurrent search worker threads
        num_insert_workers: Number of concurrent insert worker threads
        insert_batch_size: Batch size for inserts
        duration: Duration of the concurrent search+insert phase in seconds
        search_only_duration: Duration of search-only phase (no inserts) in seconds (default 0 = skip)
    """

    print("="*80)
    print("MILVUS SPACEV1B BENCHMARK")
    print("="*80)
    print(f"Configuration:")
    print(f"  Dataset path: {dataset_path}")
    print(f"  Milvus: {host}:{port}")
    print(f"  Collection: {collection_name}")
    print(f"  Initial vectors: {initial_vectors:,}")
    print(f"  Target search QPS: {search_qps:,}")
    print(f"  Target insert QPS: {insert_qps:,} vectors/sec")
    print(f"  Search workers: {num_search_workers}")
    print(f"  Insert workers: {num_insert_workers}")
    print(f"  Insert batch size: {insert_batch_size}")
    if search_only_duration > 0:
        print(f"  Search-only duration: {search_only_duration}s")
    print(f"  Concurrent phase duration: {duration}s")
    print("="*80)

    # Initialize reader
    print("\nInitializing SPACEV1B reader...")
    reader = SPACEV1BReader(dataset_path)

    # Connect to Milvus
    client = MilvusClient(uri=f"http://{host}:{port}")

    # Drop collection if exists
    if not client.has_collection(collection_name):
        print(f"\nDropping existing collection '{collection_name}'...")
        client.drop_collection(collection_name)

        # Create collection
        print(f"\nCreating collection '{collection_name}'...")
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=reader.dim)

        # Create index params
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 4096}
        )

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )

        print(f"Collection '{collection_name}' created successfully")

        # Phase 1: Load initial vectors
        load_initial_vectors(client, collection_name, reader, initial_vectors, batch_size=10000)

        # Phase 2: Flush and wait for indexing
        flush_and_index(client, collection_name, host, port)

    # Load collection into memory
    print("\nLoading collection into memory...")
    client.load_collection(collection_name)
    print("Collection loaded")

    # Read queries
    print("\nLoading query vectors...")
    queries = reader.read_queries()
    print(f"Loaded {len(queries):,} query vectors")

    # Phase 3a: Search-only phase (optional)
    search_only_stats = None
    if search_only_duration > 0:
        print(f"\n{'='*80}")
        print(f"PHASE 3A: Search-Only Baseline ({search_only_duration}s)")
        print(f"{'='*80}")
        print(f"Testing search performance WITHOUT concurrent inserts")
        print()

        # Initialize search-only latency tracker
        search_only_tracker = LatencyTracker()

        # Calculate QPS per worker for search-only
        search_qps_per_worker = search_qps / num_search_workers

        # Create search-only workers
        search_only_workers = []
        for i in range(num_search_workers):
            worker = SearchWorker(
                worker_id=i,
                client=client,
                collection_name=collection_name,
                queries=queries,
                target_qps=search_qps_per_worker,
                latency_tracker=search_only_tracker,
                duration=search_only_duration
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

            stats = search_only_tracker.get_stats()
            print(f"[{elapsed:.0f}s] Searches: {total_searches:,} ({current_qps:.0f} QPS)")
            if stats:
                print(f"        Latency - mean: {stats['mean_ms']:.2f}ms, "
                      f"p50: {stats['p50_ms']:.2f}ms, p95: {stats['p95_ms']:.2f}ms, "
                      f"p99: {stats['p99_ms']:.2f}ms")

        # Stop workers
        print("\nStopping search-only workers...")
        for worker in search_only_workers:
            worker.stop()

        # Wait for workers to finish
        for worker in search_only_workers:
            worker.join(timeout=10)

        # Collect search-only statistics
        search_only_time = time.time() - start_time
        search_only_total = sum(w.queries_executed for w in search_only_workers)
        search_only_errors = sum(w.errors for w in search_only_workers)
        search_only_stats = {
            "duration_seconds": search_only_time,
            "total_queries": search_only_total,
            "errors": search_only_errors,
            "actual_qps": search_only_total / search_only_time,
            "latency": search_only_tracker.get_stats()
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
    # print(f"\n{'='*80}")
    # print(f"PHASE {'3B' if search_only_duration > 0 else '3'}: Concurrent Search + Insert ({duration}s)")
    # print(f"{'='*80}")
    # if search_only_duration > 0:
    #     print(f"Testing search performance WITH concurrent inserts")
    #     print()

    # # Initialize latency tracker for concurrent phase
    # latency_tracker = LatencyTracker()

    # # Calculate QPS per worker
    # search_qps_per_worker = search_qps / num_search_workers
    # insert_qps_per_worker = insert_qps / num_insert_workers

    # # Calculate insert range for each worker
    # remaining_vectors = reader.num_vectors - initial_vectors
    # vectors_per_insert_worker = remaining_vectors // num_insert_workers

    # # Create search workers
    # search_workers = []
    # for i in range(num_search_workers):
    #     worker = SearchWorker(
    #         worker_id=i,
    #         client=client,
    #         collection_name=collection_name,
    #         queries=queries,
    #         target_qps=search_qps_per_worker,
    #         latency_tracker=latency_tracker,
    #         duration=duration
    #     )
    #     search_workers.append(worker)

    # # Create insert workers
    # insert_workers = []
    # for i in range(num_insert_workers):
    #     start_idx = initial_vectors + i * vectors_per_insert_worker
    #     end_idx = start_idx + vectors_per_insert_worker
    #     if i == num_insert_workers - 1:  # Last worker gets any remainder
    #         end_idx = reader.num_vectors

    #     worker = InsertWorker(
    #         worker_id=i,
    #         client=client,
    #         collection_name=collection_name,
    #         reader=reader,
    #         start_idx=start_idx,
    #         end_idx=end_idx,
    #         batch_size=insert_batch_size,
    #         target_qps=insert_qps_per_worker,
    #         duration=duration
    #     )
    #     insert_workers.append(worker)

    # # Start all workers
    # start_time = time.time()
    # for worker in search_workers + insert_workers:
    #     worker.start()

    # # Monitor progress
    # last_report = start_time
    # while time.time() - start_time < duration:
    #     time.sleep(10)

    #     elapsed = time.time() - start_time
    #     total_searches = sum(w.queries_executed for w in search_workers)
    #     total_inserts = sum(w.vectors_inserted for w in insert_workers)
    #     current_qps = total_searches / elapsed
    #     current_insert_rate = total_inserts / elapsed

    #     stats = latency_tracker.get_stats()
    #     print(f"\n[{elapsed:.0f}s] Searches: {total_searches:,} ({current_qps:.0f} QPS), "
    #           f"Inserts: {total_inserts:,} ({current_insert_rate:.0f} vec/s)")
    #     if stats:
    #         print(f"        Search latency - mean: {stats['mean_ms']:.2f}ms, "
    #               f"p50: {stats['p50_ms']:.2f}ms, p95: {stats['p95_ms']:.2f}ms, "
    #               f"p99: {stats['p99_ms']:.2f}ms")

    # # Stop all workers
    # print("\n\nStopping workers...")
    # for worker in search_workers + insert_workers:
    #     worker.stop()

    # # Wait for workers to finish
    # for worker in search_workers + insert_workers:
    #     worker.join(timeout=10)

    # # Collect final statistics
    # total_time = time.time() - start_time
    # total_searches = sum(w.queries_executed for w in search_workers)
    # total_search_errors = sum(w.errors for w in search_workers)
    # total_inserts = sum(w.vectors_inserted for w in insert_workers)
    # total_insert_errors = sum(w.errors for w in insert_workers)

    # search_stats = latency_tracker.get_stats()

    # # Print results
    # print("\n" + "="*80)
    # print("CONCURRENT PHASE RESULTS (Search + Insert)")
    # print("="*80)

    # print(f"\nDuration: {total_time:.1f}s")

    # print(f"\nSEARCH OPERATIONS:")
    # print(f"  Total queries: {total_searches:,}")
    # print(f"  Errors: {total_search_errors}")
    # print(f"  Actual QPS: {total_searches / total_time:.2f}")
    # print(f"  Target QPS: {search_qps}")
    # if search_stats:
    #     print(f"\n  Latency Distribution:")
    #     print(f"    Mean:   {search_stats['mean_ms']:8.2f} ms")
    #     print(f"    P50:    {search_stats['p50_ms']:8.2f} ms")
    #     print(f"    P80:    {search_stats['p80_ms']:8.2f} ms")
    #     print(f"    P90:    {search_stats['p90_ms']:8.2f} ms")
    #     print(f"    P95:    {search_stats['p95_ms']:8.2f} ms")
    #     print(f"    P99:    {search_stats['p99_ms']:8.2f} ms")
    #     print(f"    P99.9:  {search_stats['p99_9_ms']:8.2f} ms")
    #     print(f"    Min:    {search_stats['min_ms']:8.2f} ms")
    #     print(f"    Max:    {search_stats['max_ms']:8.2f} ms")

    # print(f"\nINSERT OPERATIONS:")
    # print(f"  Total vectors: {total_inserts:,}")
    # print(f"  Errors: {total_insert_errors}")
    # print(f"  Actual rate: {total_inserts / total_time:.2f} vectors/sec")
    # print(f"  Target rate: {insert_qps} vectors/sec")

    # # Print comparison if search-only phase was run
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
            "dataset_path": dataset_path,
            "host": host,
            "port": port,
            "collection": collection_name,
            "initial_vectors": initial_vectors,
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
            "concurrent": {
                "duration_seconds": total_time,
                "search": {
                    "total_queries": total_searches,
                    "errors": total_search_errors,
                    "actual_qps": total_searches / total_time,
                    "latency": search_stats
                },
            }
        }
    }

    filename = f"spacev1b_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {filename}")
    print("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Milvus SPACEV1B Benchmark")
    parser.add_argument("--dataset-path", default=os.path.expanduser("~/sdb/SPTAG/datasets/SPACEV1B"),
                        help="Path to SPACEV1B dataset")
    parser.add_argument("--host", default="localhost", help="Milvus host")
    parser.add_argument("--port", type=int, default=19530, help="Milvus port")
    parser.add_argument("--collection", default="spacev1b_benchmark", help="Collection name")
    parser.add_argument("--initial-vectors", type=int, default=50_000_000,
                        help="Number of vectors to load initially")
    parser.add_argument("--search-qps", type=int, default=1000,
                        help="Target search QPS")
    parser.add_argument("--insert-qps", type=int, default=10000,
                        help="Target insert rate (vectors/sec)")
    parser.add_argument("--search-workers", type=int, default=10,
                        help="Number of search worker threads")
    parser.add_argument("--insert-workers", type=int, default=5,
                        help="Number of insert worker threads")
    parser.add_argument("--insert-batch-size", type=int, default=1000,
                        help="Insert batch size")
    parser.add_argument("--search-only-duration", type=int, default=0,
                        help="Duration of search-only phase in seconds (default 0 = skip)")
    parser.add_argument("--duration", type=int, default=300,
                        help="Duration of concurrent phase (seconds)")

    args = parser.parse_args()

    run_benchmark(
        dataset_path=args.dataset_path,
        host=args.host,
        port=args.port,
        collection_name=args.collection,
        initial_vectors=args.initial_vectors,
        search_qps=args.search_qps,
        insert_qps=args.insert_qps,
        num_search_workers=args.search_workers,
        num_insert_workers=args.insert_workers,
        insert_batch_size=args.insert_batch_size,
        search_only_duration=args.search_only_duration,
        duration=args.duration
    )
