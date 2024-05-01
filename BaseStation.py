from scipy.spatial.distance import euclidean
import numpy as np
 
class BaseStation:
    def __init__(self, location):
        """
        The initialization for basestation
        :param location: the coordinate of a basestation
        """
        # controlling timeline
        self.env = None

        # include all components in our network
        self.net = None

        self.location = np.array(location)
        self.monitored_target = []
        self.direct_nodes = []