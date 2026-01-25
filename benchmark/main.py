from pymilvus import connections
connections.connect("default", host="node1", port="19530")
print("Connected to Milvus successfully!")
connections.disconnect("default")

