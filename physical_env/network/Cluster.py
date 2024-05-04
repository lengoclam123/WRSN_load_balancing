class Cluster:
    def __init__(self, id, listTargets, centroid):

        self.listTargets = listTargets 
        self.centroid = centroid
        self.id = id
        self.listNodes = []
    
