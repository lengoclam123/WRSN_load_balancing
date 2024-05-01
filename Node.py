import random
import numpy as np
from scipy.spatial.distance import euclidean
import sys
import os
sys.path.append(os.path.dirname(__file__))

class Node:

    def __init__(self, location, phy_spe):
        self.env = None
        self.net = None
    
        self.location = np.array(location)
        self.energy = phy_spe['capacity']
        self.threshold = phy_spe['threshold']
        self.capacity = phy_spe['capacity']

        self.com_range = phy_spe['com_range']
        self.sen_range = phy_spe['sen_range']
        self.prob_gp = phy_spe['prob_gp']
        self.package_size = phy_spe['package_size']
        self.er = phy_spe['er']
        self.et = phy_spe['et']
        self.efs = phy_spe['efs']
        self.emp = phy_spe['emp']

        # energRR  : replenish rate
        self.energyRR = 0

        # energyCS: consumption rate
        self.energyCS = 0

        self.id = None
        self.level = None
        self.status = 1
        self.neighbors = []
        self.listTargets = []
        self.log = []
        self.log_energy = 0
        self.check_status()

        # Edit by user

        # self.typeNode = " "
        # create inhiret 

        self.startId = None
        self.endId = None