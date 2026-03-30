# CloudLab profile (geni-lib format)
import geni.portal as portal
import geni.rspec.pg as rspec

"""
This is a profile for benchmarking vector dbs.

Instructions:
Wait for the profile instance to start, and then log in to the hosts via SSH.
Clone this repo to the /mydata directory for all hosts and run the setup scripts for each host.
"""

pc = portal.Context()

# Create a Request object to start building the RSpec.
request = pc.makeRequestRSpec()

# MinIO node - storage focused
node0 = request.RawPC("node0")
node0.hardware_type = "m400"  # or "c6525-100g" or "d750" or "c6525-25g" for better SSD
# node0.disk_image = "urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU22-64-ARM"
bs0 = node0.Blockstore("bs0", "/mydata")

node1 = request.RawPC("node1")
node1.hardware_type = "m510" # or "xl170" or "c6525-25g"
node1.disk_image = "urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU24-64-STD"
bs1 = node1.Blockstore("bs1", "/mydata")

# node2 = request.RawPC("node2")
# node2.hardware_type = "c6525-25g" # or "xl170" or "c6525-25g"
# node2.disk_image = "urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU24-64-STD"
# bs2 = node2.Blockstore("bs2", "/mydata")

# node3 = request.RawPC("node3")
# node3.hardware_type = "m510" # or "xl170" or "c6525-25g"
# node3.disk_image = "urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU24-64-STD"
# bs3 = node3.Blockstore("bs3", "/mydata")

# Milvus node - memory focused  
# node4 = request.RawPC("node4")
# node4.hardware_type = "c6620"  # Use the Optane SSD on the d750 or "c6525-100g"
# node4.disk_image = "urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU24-64-STD"
# bs4 = node4.Blockstore("bs4", "/mydata")


# Create a LAN to connect all nodes
lan = request.LAN("lan")
lan.addInterface(node0.addInterface())
lan.addInterface(node1.addInterface())
# lan.addInterface(node2.addInterface())
# lan.addInterface(node3.addInterface())
# lan.addInterface(node4.addInterface())

# Print the RSpec to the enclosing page.
pc.printRequestRSpec(request)
