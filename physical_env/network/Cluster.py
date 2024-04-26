class Cluster:
    def __init__(self, id, listTargets, centroid):

        self.listTargets = listTargets
        self.centroid = centroid
        self.cluster_id = id
        self.listNodes = None
    
