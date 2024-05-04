import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from physical_env.network.NetworkIO import NetworkIO


netIO = NetworkIO("physical_env/network/network_scenarios/bacgiang_150.yaml")
env, net = netIO.makeNetwork()



# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# from physical_env.network.Nodes.InNode import InNode

# InNode([1,2],1)