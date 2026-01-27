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
node0.hardware_type = "xl170"  # or "c6525-100g"
node0.disk_image = "urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU24-64-STD"

# Milvus node - memory focused  
node1 = request.RawPC("node1")
node1.hardware_type = "xl170"  # or "c6525-100g" or "d7615" for more RAM
node1.disk_image = "urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU24-64-STD"

# Client node
node2 = request.RawPC("node2")
node2.hardware_type = "xl170"  # or "c6525-25g"
node2.disk_image = "urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU24-64-STD"

# Create a LAN to connect all nodes
lan = request.LAN("lan")
lan.addInterface(node0.addInterface())
lan.addInterface(node1.addInterface())
lan.addInterface(node2.addInterface())

# Print the RSpec to the enclosing page.
pc.printRequestRSpec(request)
