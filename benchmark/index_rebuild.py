from pymilvus import connections, Collection, utility
import time

connections.connect(host='node1', port='19530')
coll = Collection('spacev1b')

print("Releasing collection...")
coll.release()

print("Dropping index...")
coll.drop_index()

print("Recreating index with optimized settings...")
index_params = {
    "index_type": "DISKANN",
    "metric_type": "L2",
    "params": {
        "max_degree": 70,
        "search_list_size": 128
    }
}

coll.create_index(
    field_name="embedding",
    index_params=index_params
)

print("Building index...")
start = time.time()

# Wait for index to build
while True:
    utility.wait_for_index_building_complete('spacev1b')
    print(f"Index built in {int(time.time()-start)}s")
    break

print("Loading collection...")
coll.load()
print("Done")
