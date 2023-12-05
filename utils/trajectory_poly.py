from matplotlib import legend
import numpy as np
from typing import List, Union
import matplotlib.pyplot as plt


def normalize(v):
  norm = np.linalg.norm(v)
  assert norm > 0
  return v / norm


class Polynomial:
  def __init__(self, p: Union[List[float], np.ndarray]):
    """
    Initializes a Polynomial object.

    Parameters:
    - p (list or tuple): The coefficients of the polynomial in **increasing** order.
    """
    self.p = p

  # evaluate a polynomial using horner's rule
  def eval(self, t: float) -> float:
    assert t >= 0
    x = 0.0
    for i in range(0, len(self.p)):
      x = x * t + self.p[len(self.p) - 1 - i]
    return x

  # compute and return derivative
  def derivative(self) -> 'Polynomial':
    return Polynomial([(i+1) * self.p[i+1] for i in range(0, len(self.p) - 1)])


class TrajectoryOutput:
  def __init__(self):
    self.pos    = None   # position [m]
    self.vel    = None   # velocity [m/s]
    self.acc    = None   # acceleration [m/s^2]
    self.omega  = None   # angular velocity [rad/s]
    self.yaw    = None   # yaw angle [rad]


# 4d single polynomial piece for x-y-z-yaw, includes duration.
class Polynomial4D:
  def __init__(self, duration: float, px: Union[List[float], np.ndarray], 
                 py: Union[List[float], np.ndarray], pz: Union[List[float], np.ndarray], 
                 pyaw: Union[List[float], np.ndarray]):
    self.duration = duration
    self.px       = Polynomial(px)
    self.py       = Polynomial(py)
    self.pz       = Polynomial(pz)
    self.pyaw     = Polynomial(pyaw)

  # compute and return derivative
  def derivative(self):
    return Polynomial4D(
      self.duration,
      self.px.derivative().p,
      self.py.derivative().p,
      self.pz.derivative().p,
      self.pyaw.derivative().p)

  def eval(self, t):
    result = TrajectoryOutput()
    # flat variables
    result.pos = np.array([self.px.eval(t), self.py.eval(t), self.pz.eval(t)])
    result.yaw = self.pyaw.eval(t)

    # 1st derivative
    derivative = self.derivative()
    result.vel = np.array([derivative.px.eval(t), derivative.py.eval(t), derivative.pz.eval(t)])
    dyaw = derivative.pyaw.eval(t)

    # 2nd derivative
    derivative2 = derivative.derivative()
    result.acc = np.array([derivative2.px.eval(t), derivative2.py.eval(t), derivative2.pz.eval(t)])

    # 3rd derivative
    derivative3 = derivative2.derivative()
    jerk = np.array([derivative3.px.eval(t), derivative3.py.eval(t), derivative3.pz.eval(t)])

    thrust = result.acc + np.array([0, 0, 9.81]) # add gravity

    # unit direction vector
    z_body = normalize(thrust)  #  z_acc_body
    x_world = np.array([np.cos(result.yaw), np.sin(result.yaw), 0]) # heading of nose
    y_body = normalize(np.cross(z_body, x_world)) # 
    x_body = np.cross(y_body, z_body)

    jerk_orth_zbody = jerk - (np.dot(jerk, z_body) * z_body)
    h_w = jerk_orth_zbody / np.linalg.norm(thrust)

    result.omega = np.array([-np.dot(h_w, y_body), np.dot(h_w, x_body), z_body[2] * dyaw])
    return result


class Trajectory:
  def __init__(self):
    self.polynomials = None
    self.duration = None

  def n_pieces(self):
    return len(self.polynomials)

  def loadcsv(self, filename):
    data = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=range(33))
    self.polynomials = [Polynomial4D(row[0], row[1:9], row[9:17], row[17:25], row[25:33]) for row in data]
    self.duration = np.sum(data[:,0])

  def eval(self, t, order:int=0):
    assert t >= 0
    assert t <= self.duration

    current_t = 0.0
    for p in self.polynomials:
      o = 0
      while o < order:
        p = p.derivative()
      if t <= current_t + p.duration:
        return p.eval(t - current_t)
      current_t = current_t + p.duration

  # plot the trajectory in XYZ space
  def plot_spatial(self):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for i, seg in enumerate(self.polynomials):
      t_span = np.linspace(0,seg.duration,30)
      val_x = [seg.px.eval(t) for t in t_span]
      val_y = [seg.py.eval(t) for t in t_span]
      val_z = [seg.pz.eval(t) for t in t_span]
      
      ax.plot(val_x, val_y, val_z)
      if i == 0:
        ax.scatter(val_x[0], val_y[0], val_z[0], c='r', marker='o')
      elif i == len(self.polynomials):
        ax.scatter(val_x[-1], val_y[-1], val_z[-1], c='b', marker='x')
        
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    
  # plot the trajectory over time [sec]
  def plot_temporal(self):
    fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    
    steps = 150
    t_span = np.linspace(0, self.duration, steps)
    res_list = [
        [res.pos, res.vel, res.acc]
        for t in t_span
        if (res := self.eval(t)) is not None]
    
    res = np.array(res_list)
    
    label = ['X', 'Y', 'Z']
    order_leg = ["pos", "vel", "acc"]
    for i in range(3):  # x/y/z
      for j in range(3):  # pos/vel/acc
        axs[i].plot(t_span, res[:,j,i], label=order_leg[j]) 
      axs[i].set_ylabel(label[i])
      axs[i].legend()  
      axs[i].grid()
    
    
    plt.tight_layout()
    plt.show()
      
    
      

      
        
    
