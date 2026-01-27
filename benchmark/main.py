from pymilvus import connections
import subprocess

def start():
    connections.connect("default", host="node1", port="19530")
    print("Connected to Milvus successfully!")

def get_dataset():
    try:
        subprocess.run(["./fetch_data.sh"], check=True)
        print("Fetched data sucessfully")
    except subprocess.CalledProcessError as e:
        print(f"Script failed with return code: {e.returncode}")
    except FileNotFoundError:
        print(f"Error: The script file was not found.")


def stop():
    connections.disconnect("default")

if __name__ == "__main__":
    start()
    get_dataset()
    stop()

