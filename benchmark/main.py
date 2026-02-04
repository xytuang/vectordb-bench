import os
import sys
import argparse
import subprocess

# Configuration
MILVUS_HOST = "node1"
MILVUS_PORT = "19530"
DATASET_DIR = "/mydata/spacev1b"
QUERY_VECTORS = f"{DATASET_DIR}/spacev1b_query.bin"
GROUND_TRUTH = f"{DATASET_DIR}/spacev1b_truth.bin"

def check_dataset():
    """Check if dataset files exist"""
    files = [QUERY_VECTORS, GROUND_TRUTH]
    missing = [f for f in files if not os.path.exists(f)]
    
    if missing:
        print("Dataset files missing:")
        for f in missing:
            print(f"   - {f}")
        return False
    
    print("Dataset files found:")
    return True

def download_dataset():
    """Download SPACEV1B dataset"""
    print("Downloading SPACEV1B dataset...")
    script_path = "/mydata/vectordb-bench/benchmark/fetch_data.sh"
    
    if not os.path.isfile(script_path):
        print(f"Script not found: {script_path}")
        return False
    
    try:
        subprocess.run([script_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        return False

def load_data():
    """Load vectors into Milvus"""
    print("Loading data into Milvus...")
    try:
        from loader import load_spacev1b_to_milvus
        load_spacev1b_to_milvus(
            base_dir=DATASET_DIR,
            milvus_host=MILVUS_HOST,
            milvus_port=MILVUS_PORT
        )
        return True
    except Exception as e:
        print(f"Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_benchmark():
    """Run benchmark queries"""
    print("Running benchmark...")
    try:
        from benchmark import run_benchmark
        results = run_benchmark(
            query_file=QUERY_VECTORS,
            ground_truth_file=GROUND_TRUTH,
            milvus_host=MILVUS_HOST,
            milvus_port=MILVUS_PORT
        )
        return results
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="SPACEV1B Milvus Benchmark")
    parser.add_argument("--download", action="store_true", 
                       help="Download SPACEV1B dataset")
    parser.add_argument("--load", action="store_true",
                       help="Load data into Milvus")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark queries")
    parser.add_argument("--all", action="store_true",
                       help="Run all steps (download, load, benchmark)")
    
    args = parser.parse_args()
    
    # If no args, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    print("=" * 60)
    print("SPACEV1B Milvus Benchmark")
    print("=" * 60)
    
    # Download
    if args.download or args.all:
        if not check_dataset():
            if not download_dataset():
                print("Failed to download dataset")
                return
        else:
            print("Dataset already downloaded")
    
    # Load data
    if args.load or args.all:
        if not check_dataset():
            print("Dataset not found. Run with --download first")
            return
        if not load_data():
            print("Failed to load data")
            return
    
    # Run benchmark
    if args.benchmark or args.all:
        if not check_dataset():
            print("Dataset not found. Run with --download first")
            return
        run_benchmark()

if __name__ == "__main__":
    main()
