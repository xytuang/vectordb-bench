import struct
import os
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from tqdm import tqdm

COLLECTION_NAME = "spacev1b"
DIM = 100
BATCH_SIZE = 10000

def read_spacev1b_vectors(base_dir):
    """Read SPACEV1B binary format vectors"""
    print(f"Reading vectors from {base_dir}")
    vector_files = sorted([f for f in os.listdir(base_dir) if f.startswith('spacev1b_vectors_') and f.endswith('.bin')])

    if not vector_files:
        raise FileNotFoundError(f"No vector files found in {base_dir}")

    print(f"Found {len(vector_files)} vector files: {vector_files}")


    for i in range(1, len(vector_files) + 1):
        with open(os.path.join(base_dir, 'spacev1b_vectors_%d.bin' % i), 'rb') as f:
            if i == 1:
                vec_count = struct.unpack('i', f.read(4))[0]
                vec_dimension = struct.unpack('i', f.read(4))[0]
                vecbuf = bytearray(len(vector_files) * vec_dimension)
                vecbuf_offset = 0
            while True:
                part = f.read(1048576)
                if len(part) == 0: break
                vecbuf[vecbuf_offset: vecbuf_offset + len(part)] = part
                vecbuf_offset += len(part)

        actual_vec_count = vecbuf_offset // vec_dimension
        leftover = vecbuf_offset % vec_dimension

        if leftover > 0:
            print("Found leftover bytes")

        # Convert to numpy array
        vectors = np.frombuffer(vecbuf[:vecbuf_offset], dtype=np.int8).reshape((actual_vec_count, vec_dimension))
        
        # Convert int8 to float32 for Milvus
        # Note: Milvus requires float vectors, so we normalize int8 to float
        vectors = vectors.astype(np.float32)
        
    return vectors

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
    """Main loading function"""
    # Connect to Milvus
    print(f"Connecting to Milvus at {milvus_host}:{milvus_port}")
    connections.connect("default", host=milvus_host, port=milvus_port)
    
    # Create collection
    collection = create_collection(drop_old=True)
    
    # Read all vectors
    print("Loading vectors into memory...")
    vectors = read_spacev1b_vectors(base_dir)
    total_vectors = len(vectors)
    
    print(f"Inserting {total_vectors:,} vectors into Milvus...")
    
    # Insert in batches
    num_batches = (total_vectors + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(num_batches), desc="Inserting batches"):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, total_vectors)
        
        batch_vectors = vectors[start_idx:end_idx].tolist()
        batch_ids = list(range(start_idx, end_idx))
        
        collection.insert([batch_ids, batch_vectors])
        
        # Flush periodically
        if (i + 1) % 10 == 0:
            collection.flush()
    
    # Final flush
    collection.flush()
    
    # Create index
    print("Creating index...")
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 4096}
    }
    collection.create_index("embedding", index_params)
    
    # Load collection
    collection.load()
    
    print(f"\nTotal entities: {collection.num_entities:,}")
    
    connections.disconnect("default")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python loader.py <base_vectors_file>")
        sys.exit(1)
    
    load_spacev1b_to_milvus(
        base_file=sys.argv[1],
        milvus_host="node1",
        milvus_port="19530"
    )
