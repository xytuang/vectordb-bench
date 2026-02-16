from pymilvus import connections, Collection, utility
from pymilvus.client.types import LoadState
import time

connections.connect(host='node1', port='19530')

print('Checking collection status...')
state = utility.load_state('spacev1b')
print(f'Current state: {state}')

if state != LoadState.Loaded:
    print('Collection not loaded. Loading now...')
    coll = Collection('spacev1b')
    coll.load()
    print('Waiting for load to complete...')
    
    while utility.load_state('spacev1b') != LoadState.Loaded:
        print('.', end='', flush=True)
        time.sleep(5)
    
    print('Collection loaded!')
else:
    print('Collection already loaded!')

