import casadi as ca
import numpy as np
 
from casadi import sin, cos

def shift_timestep(step_horizon, t0, state_init, u, f, crosswind=None):
    u0 = ca.DM(u)
    if crosswind is None:
        f_value = f(state_init, u0)
    else:
        f_value = f(state_init, u0) + ca.vertcat(0, 0, 0, crosswind[0], crosswind[1], 0)

    next_state = ca.DM(state_init + (step_horizon * f_value))
    
    t0 = t0 + step_horizon

    return t0, next_state, u0

def DM2Arr(dm):
    return np.array(dm.full())

def get_p_arg(state_init, ref, u_ref, v_target, k):

    ref_state = ref[k, :3]
    ref_state_next = ref[k+1, :3]

    target_state = ca.DM([ref_state[0], ref_state[1], ref_state[2], 
                          v_target*np.cos(ref_state[2]), v_target*np.sin(ref_state[2]), 0])
    
    target_state_next = ca.DM([ref_state_next[0], ref_state_next[1], ref_state_next[2],
                               v_target*np.cos(ref_state_next[2]), v_target*np.sin(ref_state_next[2]), 0])

    # print(u_ref, 'u_ref')
    p_arg_vec = ca.vertcat(state_init, target_state, target_state_next, u_ref)

    return p_arg_vec

def h_obs(state, obstacle, r, alpha):
    ox, oy = obstacle
    return 2*(state[3]*(state[0] - ox) + state[4]*(state[1] - oy)) + alpha*((ox - state[0])**2 + (oy - state[1])**2 - r**2)

def grad_h_obs(state, obstacle, alpha):
    ox, oy = obstacle
    return 2*ca.horzcat(state[3]+alpha*(state[0]-ox), state[4]+alpha*(state[1]-oy), 0, state[0] - ox, state[1] - oy, 0)

def h_ref(state, ref, w, v_target, alpha):
    x, y, vx, vy = state[0], state[1], state[3], state[4]
    
    x_ref, y_ref, theta_ref = ref[0], ref[1], ref[2]
    vx_ref, vy_ref = v_target*cos(theta_ref), v_target*sin(theta_ref)

    h0 = w**2 - ((x - x_ref)**2 + (y - y_ref)**2)
    h0_dot = 2*((x - x_ref)*(vx_ref - vx) + (y - y_ref)*(vy_ref - vy))

    return h0_dot + alpha*h0

def h_ref_dot(state, control, ref, ref_next, v_target, dt, mass):
    x, y, theta, vx, vy = state[0], state[1], state[2], state[3], state[4]
    
    x_ref, y_ref, theta_ref = ref[0], ref[1], ref[2]
    vx_ref, vy_ref = v_target*cos(theta_ref), v_target*sin(theta_ref)
    omega_ref = (ref_next[2] - ref[2])/dt
    ax_ref, ay_ref = -v_target*omega_ref*sin(theta_ref), v_target*omega_ref*cos(theta_ref)

    return 2*(-((vx_ref - vx)**2 + (vy_ref - vy)**2) 
              +((x - x_ref)*(ax_ref - control[0]*cos(theta)/mass) 
                +(y - y_ref)*(ay_ref - control[0]*sin(theta)/mass)))


class PID_CBF:
    def __init__(self, dt, v_target, 
                 F_lims, tau_lims, v_lims, omega_lims, 
                 mass, I0, r, w, R_params, alpha, safe_ref=True):
        self.safe_ref = safe_ref
        R_force, R_tau = R_params
        self.R = ca.diagcat(R_force, R_tau)

        self.alpha = alpha
        self.w = w

        # setting up PID for Force
        self.KpF = 1
        self.KiF = .5
        self.KdF = 0
        self.integralForce = 0
        self.prevForceError = 0

        # setting up PID for Tau
        self.KpT = .2
        self.KiT = 0
        self.KdT = 1
        # self.KdT = 0
        self.integralTau = 0
        self.prevTauError = 0

        self.dt = dt
        self.v_target = v_target

        self.mass = mass
        self.I0 = I0
        self.r = r # Safety radius
        
        self.F_lims = F_lims
        self.tau_lims = tau_lims
        self.v_lims = v_lims
        self.omega_lims = omega_lims

        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        vx = ca.SX.sym('vx')
        vy = ca.SX.sym('vy')
        omega = ca.SX.sym('omega')

        states = ca.vertcat(x, y, theta, vx, vy, omega)
        self.n_states = states.numel()

        force = ca.SX.sym('force')
        tau = ca.SX.sym('tau')

        controls = ca.vertcat(force, tau)
        self.n_controls = controls.numel()

        rot = ca.vertcat(
                ca.horzcat(0,  0),
                ca.horzcat(0,  0),
                ca.horzcat(0,  0),
                ca.horzcat(cos(theta)/self.mass, 0),
                ca.horzcat(sin(theta)/self.mass,  0),
                ca.horzcat(0,  1/self.I0)
            )
        
        friction = 0.3
        drift = ca.vertcat(vx, vy, omega, -vx*friction/self.mass, -vy*friction/self.mass, 0)

        RHS = drift + rot @ controls

        self.f = ca.Function('f', [states, controls], [RHS])

        self.opts = {
            'ipopt': {
                'max_iter': 2000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 0
        }

        ################################################################################
        ######### DEFINE DEFAULT SOLVER WHEN NO OBSTACLE IS DETECTED ###################

        # matrix containing the control action
        U = ca.SX.sym('U', self.n_controls)

        OPT_variables = ca.vertcat(
            U.reshape((-1, 1))
        )

        # coloumn vector for storing initial state, reference trajectory, and reference control signal
        P = ca.SX.sym('P', 3*self.n_states + self.n_controls)
        
        n = self.n_states
        con = U
        st = P[:self.n_states]
        st_ref = P[self.n_states:2*self.n_states]
        st_ref_next = P[2*self.n_states:3*self.n_states]
        con_ref = P[3*self.n_states:]
        

        cost_fn = (con.T - con_ref.T) @ self.R @ (con - con_ref)

        # Runway safety condition
        g =  (h_ref_dot(st, con, st_ref, st_ref_next, self.v_target, self.dt, self.mass) 
              + self.alpha*h_ref(st, st_ref, self.w, self.v_target, self.alpha))

        # print(g.shape)
        nlp_prob = {
            'f': cost_fn,
            'x': OPT_variables,
            'g': g,
            'p': P
        }

        self.solver_no_obs = ca.nlpsol('solver', 'ipopt', nlp_prob, self.opts)

        F_min, F_max = F_lims
        tau_min, tau_max = tau_lims

        self.lbx = ca.DM.zeros((self.n_controls, 1))
        self.ubx = ca.DM.zeros((self.n_controls, 1))

        self.lbx[0] = F_min        # force lower bound
        self.lbx[1] = tau_min    # tau lower bound 

        self.ubx[0] = F_max        # force upper bound
        self.ubx[1] = tau_max    # tau upper bound

        ubg = ca.DM.ones((1, 1))*ca.inf

        self.args_no_obs = {
            'lbg': ca.DM.zeros((1, 1)),        # constraints lower bound
            'ubg': ubg,        # constraints upper bound
            'lbx': self.lbx,
            'ubx': self.ubx
        }

    def get_u(self, state, ref, k):
        valX =  state[0]
        valY =  state[1]
        valTheta = state[2]
        valVX = state[3]
        valVY = state[4]
        
        target_x = ref[k,0]
        target_y = ref[k,1]

        actual_x = float(valX[0])
        actual_y = float(valY[0])

        current_speed = np.sqrt(valVX**2 + valVY**2)

        diff_vec = np.array([actual_x - target_x,actual_y - target_y])
        ref_vec = np.array([np.cos(ref[k,2]), np.sin(ref[k,2])])

        K_speed = 1
        K_accel = 1e-3

        ForceVelocity = K_speed*(self.v_target - current_speed) + K_accel*np.dot(diff_vec, ref_vec)

        self.integralForce += ForceVelocity * self.dt

        ForceDerivative =  (ForceVelocity - self.prevForceError)/self.dt

        self.prevForceError = ForceVelocity

        F_Control = self.KpF * ForceVelocity + self.KiF * self.integralForce + self.KdF * ForceDerivative
        # F_Control = max(self.F_lims[0], min(self.F_lims[1]+2, F_Control))

        if (actual_y-target_y)/(actual_x-target_x) > np.tan(ref[k,2]):
            direction = 1
        elif (actual_y-target_y)/(actual_x-target_x) < np.tan(ref[k,2]):
            direction = -1
        else:
            direction = 0
        
        ##### HARD CODED LOGIC, SHOULD BE REVISED ####
        if -np.pi <= ref[k,2] < -np.pi/2:
            flip = -1
        else:
            flip = 1
        ##############################################

        correction = flip*direction*np.sqrt((actual_x-target_x)**2 + (actual_y-target_y)**2)
        
        K_align = 1
        K_turn = 0.01

        TauDistance = K_align*(ref[k,2] - float(valTheta[0])) + K_turn*correction

        self.integralTau += TauDistance * self.dt

        TauDerivative =  (TauDistance - self.prevTauError)/self.dt

        self.prevTauError = TauDistance

        O_Control = self.KpT * TauDistance + self.KiT * self.integralTau + self.KdT * TauDerivative

        # O_Control = max(self.tau_lims[0], min(self.tau_lims[1], O_Control))

        u = ca.vertcat(F_Control, O_Control)

        return u
    
    def get_solver(self, obstacles=None):
        # matrix containing the control action
        U = ca.SX.sym('U', self.n_controls)

        OPT_variables = ca.vertcat(
            U.reshape((-1, 1))
        )

        # coloumn vector for storing initial state, reference trajectory, and reference control signal
        P = ca.SX.sym('P', 3*self.n_states + self.n_controls)
        
        n = self.n_states
        con = U
        st = P[:self.n_states]
        st_ref = P[self.n_states:2*self.n_states]
        st_ref_next = P[2*self.n_states:3*self.n_states]
        con_ref = P[3*self.n_states:]

        cost_fn = (con.T - con_ref.T) @ self.R @ (con-con_ref)
        # Runway safety condition
        g =  (h_ref_dot(st, con, st_ref, st_ref_next, self.v_target, self.dt, self.mass) 
              + self.alpha*h_ref(st, st_ref, self.w, self.v_target, self.alpha))
        
        fx = ca.vertcat(st[3], st[4], st[5], 0, 0, 0)
        gx = ca.vertcat(
                ca.horzcat(0,  0),
                ca.horzcat(0,  0),
                ca.horzcat(0,  0),
                ca.horzcat(cos(st[2])/self.mass, 0),
                ca.horzcat(sin(st[2])/self.mass,  0),
                ca.horzcat(0,  1/self.I0)
            )

        for obstacle in obstacles:
            gr_h_obs = grad_h_obs(st, obstacle, self.alpha)
            h_ob = h_obs(st, obstacle, self.r, self.alpha)
            g = ca.vertcat(g, gr_h_obs @ (fx + gx @ con) + self.alpha*h_ob)


        nlp_prob = {
            'f': cost_fn,
            'x': OPT_variables,
            'g': g,
            'p': P
        }

        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, self.opts)
        # the len(obstacles)+1 is to account for each obstacle condition plus the runway condition
        ubg = ca.DM.ones((len(obstacles)+1, 1))*ca.inf
            
        args = {
            'lbg': ca.DM.zeros((len(obstacles)+1, 1)),        # constraints lower bound
            'ubg': ubg,        # constraints upper bound
            'lbx': self.lbx,
            'ubx': self.ubx
        }
        # print(args)

        return solver, args
        
    def update_args(self, args, u0, state_init, ref, k):
        # print(u0, 'u0')
        args['p'] = get_p_arg(state_init, ref, u0, self.v_target, k)

        # optimization variable current state
        args['x0'] = ca.DM.zeros((self.n_controls, 1))
        # ca.vertcat(
        #     ca.reshape(u0, self.n_controls, 1)
        # )

        return args

    def get_solution(self, u0, state_init, ref, k, obstacles=None):
        if self.safe_ref:
            if obstacles:
                solver, args = self.get_solver(obstacles)
            else:
                solver = self.solver_no_obs
                args = self.args_no_obs
            

            args = self.update_args(args, u0, state_init, ref, k)
            # print('solving...')
            sol = solver(
                x0=args['x0'],
                lbx=args['lbx'],
                ubx=args['ubx'],
                lbg=args['lbg'],
                ubg=args['ubg'],
                p=args['p']
            )
            # print('got solution')
            return sol
        else:
            return {'x' : u0}
