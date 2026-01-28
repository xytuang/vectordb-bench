import struct
import os
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from tqdm import tqdm

COLLECTION_NAME = "spacev1b"
DIM = 100
BATCH_SIZE = 10000

def read_spacev1b_vectors_streaming(base_dir, batch_size=1000000):
    """
    Read SPACEV1B vectors in batches to avoid memory issues.
    Yields batches of vectors as float32 numpy arrays.
    """
    print(f"Reading vectors from {base_dir}")
    
    # List all vector files
    vector_files = sorted([f for f in os.listdir(base_dir) if f.startswith('spacev1b_vectors_') and f.endswith('.bin')])
    
    if not vector_files:
        raise FileNotFoundError(f"No vector files found in {base_dir}")
    
    print(f"Found {len(vector_files)} vector files: {vector_files}")
    
    part_count = len(vector_files)
    vec_dimension = None
    total_vecs_read = 0
    buffer = bytearray()
    
    # Read vectors file by file
    for i in range(1, part_count + 1):
        filepath = os.path.join(base_dir, 'spacev1b_vectors_%d.bin' % i)
        print(f"Processing file {i}/{part_count}: {os.path.basename(filepath)}")
        
        with open(filepath, 'rb') as fvec:
            if i == 1:
                # Read header from first file
                vec_count_header = struct.unpack('i', fvec.read(4))[0]
                vec_dimension = struct.unpack('i', fvec.read(4))[0]
                print(f"Vector dimension: {vec_dimension}")
                print(f"Header claims: {vec_count_header:,} vectors (full dataset)")
            
            # Read file in chunks and yield batches
            while True:
                chunk = fvec.read(1048576)  # 1MB chunks
                if len(chunk) == 0:
                    break
                buffer.extend(chunk)
                
                # When we have enough for a batch, yield it
                bytes_per_vector = vec_dimension
                vectors_available = len(buffer) // bytes_per_vector
                
                if vectors_available >= batch_size:
                    # Extract complete vectors for this batch
                    batch_bytes = batch_size * bytes_per_vector
                    batch_data = bytes(buffer[:batch_bytes])
                    buffer = buffer[batch_bytes:]  # Keep remainder
                    
                    # Convert to numpy and yield
                    batch_vectors = np.frombuffer(batch_data, dtype=np.int8).reshape((batch_size, vec_dimension))
                    batch_vectors = batch_vectors.astype(np.float32)
                    
                    yield batch_vectors
                    total_vecs_read += batch_size
    
    # Yield remaining vectors in buffer
    if len(buffer) > 0:
        vectors_remaining = len(buffer) // vec_dimension
        if vectors_remaining > 0:
            batch_bytes = vectors_remaining * vec_dimension
            batch_data = bytes(buffer[:batch_bytes])
            
            batch_vectors = np.frombuffer(batch_data, dtype=np.int8).reshape((vectors_remaining, vec_dimension))
            batch_vectors = batch_vectors.astype(np.float32)
            
            yield batch_vectors
            total_vecs_read += vectors_remaining
        
        leftover = len(buffer) % vec_dimension
        if leftover > 0:
            print(f"{leftover} leftover bytes at end (ignored)")
    
    print(f"Total vectors processed: {total_vecs_read:,}")

def read_spacev1b_queries(query_file):
    """Read SPACEV1B query vectors"""
    print(f"Reading queries from {query_file}")
    
    with open(query_file, 'rb') as f:
        q_count = struct.unpack('i', f.read(4))[0]
        q_dimension = struct.unpack('i', f.read(4))[0]
        
        print(f"Query count: {q_count}")
        print(f"Query dimension: {q_dimension}")
        
        queries = np.frombuffer(f.read(q_count * q_dimension), dtype=np.int8).reshape((q_count, q_dimension))
        queries = queries.astype(np.float32)
        
    return queries

def read_spacev1b_groundtruth(truth_file):
    """Read SPACEV1B ground truth"""
    print(f"Reading ground truth from {truth_file}")
    
    with open(truth_file, 'rb') as f:
        t_count = struct.unpack('i', f.read(4))[0]
        topk = struct.unpack('i', f.read(4))[0]
        
        print(f"Truth count: {t_count}")
        print(f"Top-K: {topk}")
        
        truth_vids = np.frombuffer(f.read(t_count * topk * 4), dtype=np.int32).reshape((t_count, topk))
        truth_distances = np.frombuffer(f.read(t_count * topk * 4), dtype=np.float32).reshape((t_count, topk))
        
    return truth_vids, truth_distances

def create_collection(drop_old=True):
    """Create Milvus collection"""
    if utility.has_collection(COLLECTION_NAME):
        if drop_old:
            print(f"Dropping existing collection '{COLLECTION_NAME}'")
            utility.drop_collection(COLLECTION_NAME)
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists")
            return Collection(COLLECTION_NAME)
    
    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM)
    ]
    schema = CollectionSchema(fields, description="SPACEV1B dataset")
    
    # Create collection
    collection = Collection(COLLECTION_NAME, schema)
    print(f"Created collection '{COLLECTION_NAME}'")
    
    return collection


def load_spacev1b_to_milvus(base_dir, milvus_host, milvus_port):
    """Main loading function with streaming to avoid memory issues"""
    # Connect to Milvus
    print(f"Connecting to Milvus at {milvus_host}:{milvus_port}")
    connections.connect("default", host=milvus_host, port=milvus_port)
    
    # Create collection
    collection = create_collection(drop_old=True)
    
    # Stream vectors and insert in batches
    print("Streaming vectors into Milvus...")
 
    current_id = 0
    batch_count = 0
    
    for batch_vectors in read_spacev1b_vectors_streaming(base_dir, batch_size=50000):
        num_vectors = len(batch_vectors)
        batch_ids = list(range(current_id, current_id + num_vectors))
        
        # Insert batch
        collection.insert([batch_ids, batch_vectors.tolist()])
        
        current_id += num_vectors
        batch_count += 1
        
        # Flush periodically
        if batch_count % 5 == 0:
            collection.flush()
            print(f"Inserted {current_id:,} vectors so far...")
    
    # Final flush
    collection.flush()
    print(f"Inserted {current_id:,} vectors total")
    
    # Create index
    print("Creating index...")
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 4096}
    }
    collection.create_index("embedding", index_params)
    print("Index created")
    
    # Load collection
    print("Loading collection to memory...")
    collection.load()
    print("Collection loaded to memory")
    
    print(f"Total entities: {collection.num_entities:,}")
 
    connections.disconnect("default")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python loader.py <base_vectors_file>")
        sys.exit(1)
    
    load_spacev1b_to_milvus(
        base_dir=sys.argv[1],
        milvus_host="node1",
        milvus_port="19530"
    )
