from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import numpy as np

# Connect to Milvus
print("Connecting to Milvus at node1:19530...")
connections.connect(host="node1", port="19530")
print("Connected to Milvus")

# Define collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
]
schema = CollectionSchema(fields=fields, description="Test collection")

# Create collection
print("Creating test collection...")
collection = Collection(name="test_collection", schema=schema)
print("Collection created")

# Insert test data
print("Inserting test vectors...")
vectors = np.random.rand(1000, 128).tolist()
entities = [vectors]
collection.insert(entities)
print("Inserted 1000 vectors")

# Flush to ensure data is written to MinIO
print("Flushing data to storage...")
collection.flush()
print("Data flushed")

# Create index
print("Creating index...")
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)
print("Index created")

print("Test completed successfully!")
print("Now check MinIO for data...")
