import casadi as ca
import numpy as np
 
from .mpc_params import *
from casadi import sin, cos

def shift_timestep(step_horizon, t0, state_init, u, f):
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    t0 = t0 + step_horizon
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    return t0, next_state, u0

def DM2Arr(dm):
    return np.array(dm.full())

def get_p_arg(state_init, ref, k):
    p_arg_vec = ca.vertcat(state_init)
    for l in range(N):
        if k+l < len(ref):
            ref_state = ref[k+l, :2]
            if l == 0:
                target_theta = np.arctan2(ref_state[1] - state_init[1], ref_state[0] - state_init[0])
            else:
                target_theta = np.arctan2(ref_state[1] - ref[k+l-1,1], ref_state[0] - ref[k+l-1,0])
            v_command = v_target 
        else:
            ref_state = ref[-1, :2]
            target_theta = np.arctan2(ref_state[1] - ref[-2,1], ref_state[0] - ref[-2,0])
            v_command = 0
        
        target_state = ca.DM([ref_state[0], ref_state[1], target_theta, v_command*np.cos(target_theta), v_command*np.sin(target_theta), 0])
        
        p_arg_vec = ca.vertcat(p_arg_vec, target_state)
    return p_arg_vec

x = ca.SX.sym('x')
y = ca.SX.sym('y')
theta = ca.SX.sym('theta')
vx = ca.SX.sym('vx')
vy = ca.SX.sym('vy')
omega = ca.SX.sym('omega')

states = ca.vertcat(x, y, theta, vx, vy, omega)
n_states = states.numel()

force = ca.SX.sym('force')
tau = ca.SX.sym('tau')

controls = ca.vertcat(force, tau)
n_controls = controls.numel()

# matrix containing all states over all time steps +1 (each column is a state vector)
X = ca.SX.sym('X', n_states, N + 1)

# matrix containing all control actions over all time steps (each column is an action vector)
U = ca.SX.sym('U', n_controls, N)

# coloumn vector for storing initial state and reference trajectory
P = ca.SX.sym('P', n_states + N*n_states)

# state weights matrix (Q_X, Q_Y, Q_THETA)
Q = ca.diagcat(Q_x, Q_y, Q_theta, 
               Q_vx, Q_vy, Q_omega)

# controls weights matrix
R = ca.diagcat(R1, R2)

rot = ca.vertcat(
    ca.horzcat(0,  0),
    ca.horzcat(0,  0),
    ca.horzcat(0,  0),
    ca.horzcat(cos(theta)/mass, 0),
    ca.horzcat(sin(theta)/mass,  0),
    ca.horzcat(0,  1/I0)
)

drift = ca.vertcat(vx, vy, omega, 0, 0, 0)

RHS = drift + rot @ controls

f = ca.Function('f', [states, controls], [RHS])

cost_fn = 0  # cost function
g = X[:, 0] - P[:n_states]  # constraints in the equation


# runge kutta
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    cost_fn = cost_fn \
        + (st - P[(k+1)*n_states:(k+2)*n_states]).T @ Q @ (st - P[(k+1)*n_states:(k+2)*n_states]) \
        + con.T @ R @ con
    st_next = X[:, k+1]
    k1 = f(st, con)
    k2 = f(st + dt/2*k1, con)
    k3 = f(st + dt/2*k2, con)
    k4 = f(st + dt * k3, con)
    st_next_RK4 = st + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    g = ca.vertcat(g, st_next - st_next_RK4)


OPT_variables = ca.vertcat(
    X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
    U.reshape((-1, 1))
)
nlp_prob = {
    'f': cost_fn,
    'x': OPT_variables,
    'g': g,
    'p': P
}

opts = {
    'ipopt': {
        'max_iter': 2000,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))

lbx[0: n_states*(N+1): n_states] = -ca.inf     # X lower bound
lbx[1: n_states*(N+1): n_states] = -ca.inf     # Y lower bound
lbx[2: n_states*(N+1): n_states] = -ca.inf     # theta lower bound
lbx[3: n_states*(N+1): n_states] = vx_min     # VX lower bound
lbx[4: n_states*(N+1): n_states] = vy_min     # VY lower bound
lbx[5: n_states*(N+1): n_states] = omega_min     # omega lower bound

ubx[0: n_states*(N+1): n_states] = ca.inf      # X upper bound
ubx[1: n_states*(N+1): n_states] = ca.inf      # Y upper bound
ubx[2: n_states*(N+1): n_states] = ca.inf      # theta upper bound
ubx[3: n_states*(N+1): n_states] = vx_max     # VX upper bound
ubx[4: n_states*(N+1): n_states] = vy_max      # VY upper bound
ubx[5: n_states*(N+1): n_states] = omega_max      # omega upper bound

lbx[n_states*(N+1)::2] = F_min                 # force lower bound
lbx[n_states*(N+1)+1::2] = tau_min             # tau lower bound 

ubx[n_states*(N+1)::2] = F_max                  # v upper bound for all V
ubx[n_states*(N+1)+1::2] = tau_max                  # v upper bound for all V


args = {
    'lbg': ca.DM.zeros((n_states*(N+1), 1)),  # constraints lower bound
    'ubg': ca.DM.zeros((n_states*(N+1), 1)),  # constraints upper bound
    'lbx': lbx,
    'ubx': ubx
}
