import sys
sys.path.append('..')

import numpy as np
from utils import trajectory as traj
from pycrazyswarm import *

if __name__ == "__main__":
    # swarm = Crazyswarm()
    # timeHelper = swarm.timeHelper
    # allcfs = swarm.allcfs

    traj1 = traj.Trajectory()
    traj1.loadcsv("data_base/figure8.csv")
    traj1.plot_spatial()







