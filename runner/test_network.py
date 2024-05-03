import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from physical_env.network.NetworkIO import NetworkIO


def log(net, mcs):
    
    while True:
        print(net.env.now, net.check_nodes())
        yield net.env.timeout(1.0)


netIO = NetworkIO("physical_env/network/network_scenarios/bacgiang_50.yaml")
env, net = netIO.makeNetwork()

x = env.process(net.operate())
env.process(log(net, None))

env.run(until=x)