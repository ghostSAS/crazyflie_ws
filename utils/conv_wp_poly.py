"""
if the waypoints are dense in terms of time, rapaidly call goTo() function (rate >> 1 hz) may lead to instability
therefore, waypoints are converted to a collection of polynomial4ds first, then imploy
    - startTrajectory(): finish the trajectory at the exact set period
    - cmdFullState(): finish the trajectory ASAP
    
TODO
    - test if cmdPosition() is stable for tracking dense waypoints
    - add yaw into the converted polynomial4d, yaw_k = arctan(dy_k, dx_k), s.t. abs(dyaw_k) <= threshold
"""
   
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt



class T_vec:
    def __init__(self, order:int) -> None:
        self.order = order
        
    def eval(self, t:float) -> ca.DM:
        x = [1.0]
        for i in range(self.order):
            x.append(x[-1]*t)
        return ca.DM(x).T

    def derivative(self, t:float, order:int) -> ca.DM:
        base = self.eval(t)
        assert t>= 0
        
        if order == 0:
            return base
        
        for o in range(order):
            tmp = ca.DM.zeros(1,base.shape[1])
            for i in range(base.shape[1]-1):
                tmp[i+1] = base[i]*(i+1)
            base = tmp
            
        return base


class Convert_waypoints_to_poly():
    # wp: N*4 ndarray
    def __init__(self, wp:np.ndarray, seg_num:int=3, poly_order:int=7):
        assert wp.shape[1] == 4
        
        self.dt = wp[1,0] - wp[0,0] # length of time step
        if self.dt >= .5:
            raise ValueError("the given waypoints are too sparse, consider directly using goTo()")
        self.seg_num = seg_num  # evenlly divide the waypoints to the number of segments
        self.poly_order = poly_order
        self.vel_bound = 1.5*2
        self.acc_bound = 5*2
        
        self.setup_wp(wp)
        
    
    def setup_wp(self, wp):
        wp -= wp[0,:]
        self.wp = []    # col 0->3: time, x, y, z
       
        seg_len_idx = wp.shape[0]//self.seg_num
        for s in range(self.seg_num):
            if s != self.seg_num-1:
                tmp = wp[s*seg_len_idx: (s+1)*seg_len_idx+1, :]
            else:
                tmp = wp[s*seg_len_idx:, :]
            tmpt = tmp[:,0].copy()
            tmpt -= tmpt[0]
            self.wp.append(np.c_[tmpt,tmp[:,1:]])      
        
    
    # setup optimaization for     
    def setup_opti(self, ax:str):
        col = 0
        if ax=='x':
            col = 1
        elif ax=='y':
            col = 2
        elif ax=='z':
            col = 3
        else:
            raise ValueError("wrong ax, choose string from x y z")
            
        opti = ca.Opti()
            
        N = self.seg_num * (self.poly_order+1)    # number of variables
        P = opti.variable(N,1)    # decision variable
        t_vec = T_vec(self.poly_order)
        
        # -------- construct cost functional ----------
        # self.A = np.zeros([self.wp.shape[0],self.poly_order*self.num_seg])
        J = 0
        for s, wp in enumerate(self.wp):
            for n in range(wp.shape[0]):
                P_idx = np.arange(s*(self.poly_order+1), (s+1)*(self.poly_order+1))
                J += (t_vec.eval(wp[n,0])@P[P_idx] - wp[n,col])**2
        
        opti.minimize(J)
        
        # ------- constuct constrains ----------
        # boundry constrain
        P_idx = np.arange(0, self.poly_order+1)
        opti.subject_to(t_vec.eval(0)@P[P_idx] == 0)            # pos starts from 0
        opti.subject_to(t_vec.derivative(0,1)@P[P_idx] == 0)    # vel starts from 0
        opti.subject_to(t_vec.derivative(0,2)@P[P_idx] == 0)    # acc starts from 0
        
        t_end = self.wp[-1][-1,0]
        P_idx = np.arange((self.poly_order+1)*(self.seg_num-1), N)
        opti.subject_to(t_vec.derivative(t_end,1)@P[P_idx] == 0)    # vel ends at 0
        opti.subject_to(t_vec.derivative(t_end,2)@P[P_idx] == 0)    # acc ends at 0
        
        # continuity
        for i in range(self.seg_num-1):
            t_end = self.wp[i][-1,0]
            P0_idx = np.arange((self.poly_order+1)*i, (self.poly_order+1)*(i+1))
            P1_idx = np.arange((self.poly_order+1)*(i+1), (self.poly_order+1)*(i+2))
            
            opti.subject_to(t_vec.eval(t_end)@P[P0_idx] == t_vec.eval(0)@P[P1_idx])     # pos continue
            opti.subject_to(t_vec.derivative(t_end,1)@P[P0_idx] == t_vec.derivative(0,1)@P[P1_idx])     # vel continue
            opti.subject_to(t_vec.derivative(t_end,2)@P[P0_idx] == t_vec.derivative(0,2)@P[P1_idx])     # acc continue
            
        # restriction on vel and acc (or pos)
        for i, s in enumerate(self.wp):
            P_idx = np.arange((self.poly_order+1)*i, (self.poly_order+1)*(i+1))
            for t in s[:,0]:
                vel = t_vec.derivative(t,1)@P[P_idx]
                opti.subject_to(-self.vel_bound <= vel)
                opti.subject_to(vel <= self.vel_bound)
                
                acc = t_vec.derivative(t,2)@P[P_idx]
                opti.subject_to(-self.acc_bound <= acc)
                opti.subject_to(acc <= self.acc_bound)
                
        opti.set_initial(P,np.ones((N,1)))

        return opti, P
    
    
    def convert(self, print_level:int=0):
        poly_table = np.zeros((self.seg_num, (self.poly_order+1)*4+1))
        poly_table[:,0] = [self.wp[i][-1,0] for i in range(self.seg_num)]
        
        opts_dict = dict()
        opts_dict["ipopt.print_level"] = print_level
        opts_dict["ipopt.sb"] = "yes"
        opts_dict["print_time"] = print_level
        axes = ['x','y','z']
        for i in range(len(axes)):
            opti, P_var = self.setup_opti(axes[i])
            opti.solver('ipopt', opts_dict)
            P = opti.solve().value(P_var).reshape((self.seg_num, self.poly_order+1))
            poly_table[:,1+(self.poly_order+1)*i:1+(self.poly_order+1)*(i+1)] = P
            
        self.poly_table = poly_table
        
        
    def plot(self, wp):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        t_vec = T_vec(self.poly_order)
        steps = 100
        axes = ['x','y','z']
        
        for row in self.poly_table:
            t_span = np.linspace(0,row[0],steps)
            XYZ = np.zeros((steps, 3))
            for i in range(len(axes)):
                P = row[1+(self.poly_order+1)*i:1+(self.poly_order+1)*(i+1)]
                tmp = np.array([t_vec.eval(t)@P for t in t_span]).squeeze().squeeze()
                XYZ[:,i] = tmp
            ax.plot(XYZ[:,0], XYZ[:,1], XYZ[:,2], 'r', linewidth=2.0)
        
        ax.scatter(wp[:,1], wp[:,2], wp[:,3], s=5, c='b')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.show()
        
        
    def to_csv(self, file_name):
        csv_filename = '../data_base/'+file_name
        # file_exists = os.path.isfile(csv_filename)
        # with open(csv_filename, 'a' if file_exists else 'w', newline='') as csvfile:
        #     csv_writer = csv.writer(csvfile)
            
        #     # Write header
        header = "duration," + ",".join([f"x^{i}" for i in range(8)] + \
                [f"y^{i}" for i in range(8)] + [f"z^{i}" for i in range(8)] + \
                [f"yaw^{i}" for i in range(8)])

        #     csv_writer.writerow(header)
            
        #     # Write data
        #     for row in self.poly_table:
        #         # csv_writer.writerow(row)
        #         np.

        np.savetxt(csv_filename, self.poly_table, \
            header=header, delimiter= ",", comments="", fmt='%.6f')
        print(f'Data has been written to {csv_filename}')
        
        return 0
            