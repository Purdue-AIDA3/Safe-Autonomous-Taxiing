import casadi as ca
import numpy as np
 
from casadi import sin, cos

def shift_timestep(step_horizon, t0, state_init, u, f, crosswind=None):
    if crosswind is None:
        f_value = f(state_init, u[:, 0])
    else:
        f_value = f(state_init, u[:, 0]) + ca.vertcat(0, 0, 0, crosswind[0], crosswind[1], 0)
    
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    t0 = t0 + step_horizon
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    return t0, next_state, u0

def DM2Arr(dm):
    return np.array(dm.full())

def get_p_arg(state_init, ref, v_target, k, N):
    p_arg_vec = ca.vertcat(state_init)
    for l in range(N):
        if k+l < len(ref):
            ref_state = ref[k+l, :3]
            v_command = v_target 
        else:
            ref_state = ref[-1, :3]
            v_command = 0

        target_state = ca.DM([ref_state[0], ref_state[1], ref_state[2], v_command*np.cos(ref_state[2]), v_command*np.sin(ref_state[2]), 0])
        
        p_arg_vec = ca.vertcat(p_arg_vec, target_state)

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


class MPC_CBF:
    def __init__(self, dt, v_target, 
                 Q_params, R_params, 
                 F_lims, tau_lims, v_lims, omega_lims, 
                 N, mass, I0, r, w, alpha, safe_ref=True):
        self.dt = dt
        self.v_target = v_target

        self.mass = mass
        self.I0 = I0
        self.N = N # MPC time window length
        self.r = r # Safety radius
        self.w = w # Trajectory tracking radius
        self.alpha = alpha # Class-K function parameter, must be positive
        
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
        
        # state weights matrix (Q_X, Q_Y, Q_THETA)
        Q_x, Q_y, Q_theta, Q_vx, Q_vy, Q_omega = Q_params
        self.Q = ca.diagcat(Q_x, Q_y, Q_theta, Q_vx, Q_vy, Q_omega)

        # controls weights matrix
        R_force, R_tau = R_params
        self.R = ca.diagcat(R_force, R_tau)

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

        # matrix containing all states over all time steps +1 (each column is a state vector)
        X = ca.SX.sym('X', self.n_states, self.N + 1)

        # matrix containing all control actions over all time steps (each column is an action vector)
        U = ca.SX.sym('U', self.n_controls, self.N)

        OPT_variables = ca.vertcat(
            X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
            U.reshape((-1, 1))
        )

        # coloumn vector for storing initial state and reference trajectory
        P = ca.SX.sym('P', self.n_states + self.N*self.n_states)
        
        cost_fn = 0  # cost function
        g = X[:, 0] - P[:self.n_states]  # constraints in the equation

        
        n = self.n_states
        for k in range(self.N):
            st = X[:, k]
            con = U[:, k]
            ref_k = P[(k+1)*n:(k+2)*n]

            if k != self.N-1:
                ref_k_next = P[(k+2)*n:(k+3)*n]
            else:
                ref_k_next = ref_k
            
            cost_fn = cost_fn \
                + (st - ref_k).T @ self.Q @ (st - ref_k) \
                + con.T @ self.R @ con
            st_next = X[:, k+1]
            k1 = self.f(st, con)
            k2 = self.f(st + dt/2*k1, con)
            k3 = self.f(st + dt/2*k2, con)
            k4 = self.f(st + dt * k3, con)
            st_next_RK4 = st + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            g = ca.vertcat(g, st_next - st_next_RK4)
            # Runway safety condition
            if safe_ref:
                g = ca.vertcat(g, h_ref_dot(st, con, ref_k, ref_k_next, self.v_target, self.dt, self.mass) 
                                + self.alpha*h_ref(st, ref_k, self.w, self.v_target, self.alpha))


        nlp_prob = {
            'f': cost_fn,
            'x': OPT_variables,
            'g': g,
            'p': P
        }

        self.solver_no_obs = ca.nlpsol('solver', 'ipopt', nlp_prob, self.opts)

        F_min, F_max = F_lims
        tau_min, tau_max = tau_lims

        v_min, v_max = v_lims
        omega_min, omega_max = omega_lims

        self.lbx = ca.DM.zeros((self.n_states*(self.N+1) + self.n_controls*self.N, 1))
        self.ubx = ca.DM.zeros((self.n_states*(self.N+1) + self.n_controls*self.N, 1))

        self.lbx[0: self.n_states*(self.N+1): self.n_states] = -ca.inf     # X lower bound
        self.lbx[1: self.n_states*(self.N+1): self.n_states] = -ca.inf     # Y lower bound
        self.lbx[2: self.n_states*(self.N+1): self.n_states] = -ca.inf     # theta lower bound
        self.lbx[3: self.n_states*(self.N+1): self.n_states] = v_min       # VX lower bound
        self.lbx[4: self.n_states*(self.N+1): self.n_states] = v_min       # VY lower bound
        self.lbx[5: self.n_states*(self.N+1): self.n_states] = omega_min   # omega lower bound

        self.ubx[0: self.n_states*(self.N+1): self.n_states] = ca.inf      # X upper bound
        self.ubx[1: self.n_states*(self.N+1): self.n_states] = ca.inf      # Y upper bound
        self.ubx[2: self.n_states*(self.N+1): self.n_states] = ca.inf      # theta upper bound
        self.ubx[3: self.n_states*(self.N+1): self.n_states] = v_max       # VX upper bound
        self.ubx[4: self.n_states*(self.N+1): self.n_states] = v_max       # VY upper bound
        self.ubx[5: self.n_states*(self.N+1): self.n_states] = omega_max   # omega upper bound

        self.lbx[self.n_states*(self.N+1)::self.n_controls] = F_min        # force lower bound
        self.lbx[self.n_states*(self.N+1)+1::self.n_controls] = tau_min    # tau lower bound 

        self.ubx[self.n_states*(self.N+1)::self.n_controls] = F_max        # force upper bound
        self.ubx[self.n_states*(self.N+1)+1::self.n_controls] = tau_max    # tau upper bound

        if safe_ref:
            ubg = ca.DM.zeros((self.n_states*(self.N+1)+self.N, 1))

            ubg[0:2*n] = 0
            
            ubg[2*n ::n+1] = ca.inf 
            # ubg[2*n ::n+1] = 0 

            self.args_no_obs = {
                'lbg': ca.DM.zeros((self.n_states*(self.N+1)+self.N, 1)),        # constraints lower bound
                'ubg': ubg,        # constraints upper bound
                'lbx': self.lbx,
                'ubx': self.ubx
            }
        else:
            self.args_no_obs = {
                'lbg': ca.DM.zeros((self.n_states*(self.N+1), 1)),        # constraints lower bound
                'ubg': ca.DM.zeros((self.n_states*(self.N+1), 1)),        # constraints upper bound
                'lbx': self.lbx,
                'ubx': self.ubx
            }



    def get_solver(self, obstacles=None):
        # matrix containing all states over all time steps +1 (each column is a state vector)
        X = ca.SX.sym('X', self.n_states, self.N + 1)

        # matrix containing all control actions over all time steps (each column is an action vector)
        U = ca.SX.sym('U', self.n_controls, self.N)

        OPT_variables = ca.vertcat(
            X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
            U.reshape((-1, 1))
        )

        # coloumn vector for storing initial state and reference trajectory
        P = ca.SX.sym('P', self.n_states + self.N*self.n_states)
        
        cost_fn = 0  # cost function
        g = X[:, 0] - P[:self.n_states]  # constraints in the equation

        
        n = self.n_states
        for k in range(self.N):
            st = X[:, k]
            con = U[:, k]
            ref_k = P[(k+1)*n:(k+2)*n]

            if k != self.N-1:
                ref_k_next = P[(k+2)*n:(k+3)*n]
            else:
                ref_k_next = ref_k
            
            cost_fn = cost_fn \
                + (st - ref_k).T @ self.Q @ (st - ref_k) \
                + con.T @ self.R @ con
            st_next = X[:, k+1]
            k1 = self.f(st, con)
            k2 = self.f(st + self.dt/2*k1, con)
            k3 = self.f(st + self.dt/2*k2, con)
            k4 = self.f(st + self.dt * k3, con)
            st_next_RK4 = st + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            g = ca.vertcat(g, st_next - st_next_RK4)
            # Runway safety condition
            g = ca.vertcat(g, h_ref_dot(st, con, ref_k, ref_k_next, self.v_target, self.dt, self.mass) 
                           + self.alpha*h_ref(st, ref_k, self.w, self.v_target, self.alpha))
            
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
        ubg = ca.DM.zeros((self.n_states*(self.N+1)+(len(obstacles)+1)*self.N, 1))

        ubg[0:2*n] = 0
        for o in range(len(obstacles)+1):
            ubg[2*n + o::(len(obstacles)+1)+n] = ca.inf
            
        args = {
            'lbg': ca.DM.zeros((self.n_states*(self.N+1)+(len(obstacles)+1)*self.N, 1)),        # constraints lower bound
            'ubg': ubg,        # constraints upper bound
            'lbx': self.lbx,
            'ubx': self.ubx
        }

        return solver, args

    def update_args(self, args, X0, u0, state_init, ref, k):
        args['p'] = get_p_arg(state_init, ref, self.v_target, k, self.N)

        # optimization variable current state
        args['x0'] = ca.vertcat(
            ca.reshape(X0, self.n_states*(self.N+1), 1),
            ca.reshape(u0, self.n_controls*self.N, 1)
        )

        return args

    def get_solution(self, X0, u0, state_init, ref, k, obstacles=None):
        if obstacles:
            solver, args = self.get_solver(obstacles)
        else:
            solver = self.solver_no_obs
            args = self.args_no_obs

        args = self.update_args(args, X0, u0, state_init, ref, k)
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
