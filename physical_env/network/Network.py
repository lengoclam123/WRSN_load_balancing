from physical_env.network.utils.clustering.kmean_plus_plus import clustering
from physical_env.network.utils.create_edges.basic import createEdges
from physical_env.network.utils.create_node_in_cluster.basic import createNodeInCluster
from physical_env.network.utils.create_node_between_cluster.basic import createNodeBetweenCluster
import warnings
warnings.filterwarnings("ignore")

class Network:
    def __init__(self, env, baseStation, listTargets ,max_time,phy):
        self.env = env
        self.baseStation = baseStation
        self.listTargets = listTargets
        self.phy = phy
        self.targets_active = [1 for _ in range(len(self.listTargets))]
        self.alive = 1
        self.found = []
        self.Alpha = 0.8
        baseStation.env = self.env
        baseStation.net = self
        self.max_time = max_time
        it = 0

        for target in listTargets:
            target.id = it
            it += 1
        
        self.listNodes = []
        self.listClusters = []
        self.listEdges = []
        self.createNodes()
        for cluster in self.listClusters:
            for node in cluster.listNodes:
                    node.cluster_id = cluster.id
    
    def createNodes(self):
        self.listClusters = clustering(self)
        self.listEdges = createEdges(self)
        nodeInsideCluster = createNodeInCluster(self)
        nodeBetweenCluster = createNodeBetweenCluster(self)
        self.listNodes = nodeBetweenCluster + nodeInsideCluster

    def setLevels(self):
        for node in self.listNodes:
            node.level = -1
        tmp1 = []
        tmp2 = []
        for node in self.baseStation.direct_nodes:
            if node.status == 1:
                node.level = 1
                tmp1.append(node)

        for i in range(len(self.targets_active)):
            self.targets_active[i] = 0

        while True:
            if len(tmp1) == 0:
                break
            # For each node, we set value of target covered by this node as 1
            # For each node, if we have not yet reached its neighbor, then level of neighbors equal this node + 1
            for node in tmp1:
                #print(node.location[0],node.location[1])
                for target in node.listTargets:
                    self.targets_active[target.id] = 1
                for neighbor in node.neighbors:
                    if neighbor.status == 1 and neighbor.level == -1:
                        tmp2.append(neighbor)
                        neighbor.level = node.level + 1

            # Once all nodes at current level have been expanded, move to the new list of next level
            tmp1 = tmp2[:]
            tmp2.clear()
        return

    def operate(self, t=1):
        for node in self.listNodes:
            self.env.process(node.operate(t=t))
            self.env.process(self.baseStation.operate(t=t))
        while True:
            yield self.env.timeout(t / 10.0)
            self.setLevels()
            self.alive = self.check_targets()
            yield self.env.timeout(9.0 * t / 10.0)
            if self.alive == 0 or self.env.now >= self.max_time:
                print("die")
                break         
        return

    def check_targets(self):
        return min(self.targets_active)
    
    def check_nodes(self):
        tmp = 0
        for node in self.listNodes:
            if node.status == 0:
                tmp += 1
        return tmp
    
    def get_dead_nodes(self):
        list_dead_nodes = []
        for node in self.listNodes:
            if node.status == 0:
                list_dead_nodes.append(node)
        return list_dead_nodes
