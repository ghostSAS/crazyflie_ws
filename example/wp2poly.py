import sys

sys.path.append('/home/ykgroup/crazyswarm/ros_ws/src/crazyswarm/scripts/crazyflie_ws')
sys.path.append('/home/ykgroup/crazyswarm/ros_ws/src/crazyswarm/scripts')

import numpy as np
from utils import conv_wp_poly as conv
import matplotlib.pyplot as plt

# construct waypoints
def create_waypoints():
    dt = .05
    N = 200
    t_span = np.linspace(0,dt*(N-1),N)

    wp = np.zeros((N,4))

    wp[:,0] = t_span

    R = 1.5
    w = 1/3
    wp[:,1] = R*np.sin(t_span*w)
    wp[:,2] = R*np.cos(t_span*w)
    wp[:,3] = 1*np.sin(t_span*3)
    return wp

def plot_waypoints(wp):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting
    ax.plot(wp[:, 1], wp[:, 2], wp[:, 3])

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()


# convert waypoints to polynomials
def plot_time_idx(ts):
    for t in ts:
        plt.plot(t)
        
    plt.show()
    
wp = create_waypoints()
conv_poly = conv.Convert_waypoints_to_poly(wp, seg_num=4)

# this should plot the same as the previous one
# with shifted coordiante
plot_waypoints(conv_poly.wp)

# you are supposed to see one straight increasing line 
# multiple lines are overlapped 
plot_time_idx([w[:,0] for w in conv_poly.wp])

# run optimization
conv_poly.convert()

# print(conv_poly.poly_table)
conv_poly.plot(wp)

# write the computed polynomial coefficients to csv file
conv_poly.to_csv('test.csv')

