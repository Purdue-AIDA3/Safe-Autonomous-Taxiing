import casadi as ca
import numpy as np
import src.params as params

import src.mpc_casadi as mpc
from src.mpc_casadi import MPC_CBF

import src.pid_casadi as pid
from src.pid_casadi import PID_CBF

from time import time
from src.animation_code import simulate
from src.utils import gen_waypoints, gen_reference_trajectory
from src.taxiway_graph import TaxiwayGraph

if __name__ == '__main__':
    csv_file_path = r".\data\Position_of_Airport.csv"
    image_path = r".\data\map.png"

    # Create the TaxiwayGraph object
    taxiway_graph = TaxiwayGraph(csv_file_path, image_path)

    # Define the start and end gates
    start_gate = 'Gate 2'
    end_runway = 'R10L'

    # Find the shortest path from the starting gate to the runway
    shortest_path_flight, control_points = taxiway_graph.find_shortest_path_flight(start_gate, end_runway)

    turning_radius = 5
    turning_res = 10

    waypoints = gen_waypoints(control_points, turning_radius, turning_res)

    ref = gen_reference_trajectory(waypoints, params.v_target, params.dt)

    Q_params = [params.Q_x, params.Q_y, params.Q_theta, params.Q_vx, params.Q_vx, params.Q_omega]
    R_params = [params.R1, params.R2]
    F_lims = [params.F_min, params.F_max]
    tau_lims = [params.tau_min, params.tau_max]
    v_lims = [params.v_min, params.v_max]
    omega_lims = [params.omega_min, params.omega_max]

    obs_inds = [150, 400]
    obstacles0 = [(ref[obs_inds[0],0]-9, ref[obs_inds[0],1]),
                (ref[obs_inds[1],0]-9, ref[obs_inds[1],1])]
    

    r = 10 # Safety radius
    w = 8 # Runway radius

    ####### Method,  safe,  obs,  wind ########
    tests = [
            #  ('PID', False, False, False),
            #  ('PID', True, False, False),
            #  ('PID', True, False, True),
            #  ('PID', True, True, False),
            #  ('PID', True, True, True),
            #  ('MPC', False, False, False),
            #  ('MPC', True, False, False),
            #  ('MPC', True, False, True),
             ('MPC', True, True, False),
            #  ('MPC', True, True, True),
            #  ('MPC', False, False, True)
             ]

    results = []

    for test in tests:
        x_init = ref[0,0]
        y_init = ref[0,1]
        theta_init = ref[0,2]
        state_init = ca.DM([x_init, y_init, theta_init, 0, 0, 0])        # initial state
        
        print(test)
        method, safe, obs, wind = test

        if method == 'PID':
            R_params = [1, 1]
            alpha = 1 # Parameter for scalar class-K function, must be positive

            method_cbf = PID_CBF(params.dt, params.v_target, 
                            F_lims, tau_lims, v_lims, omega_lims, 
                            params.mass, params.I0, r, w, R_params, 
                            alpha, safe_ref=safe)
            
            u0 = ca.DM.zeros((method_cbf.n_controls, 1))      # initial control

            cat_states = mpc.DM2Arr(state_init)
            cat_controls = mpc.DM2Arr(u0)

        elif method == 'MPC':
            alpha = 10 # Parameter for scalar class-K function, must be positive
            method_cbf = MPC_CBF(params.dt, params.v_target, 
                                Q_params, R_params, 
                                F_lims, tau_lims, v_lims, omega_lims,
                                params.N, params.mass, params.I0, 
                                r, w, alpha, safe_ref=safe)
            
            u0 = ca.DM.zeros((method_cbf.n_controls, params.N))      # initial control
            X0 = ca.repmat(state_init, 1, params.N+1)         # initial state full
            mpc_iter = 0
            cat_states = mpc.DM2Arr(X0)
            cat_controls = mpc.DM2Arr(u0[:, 0])
            
        if obs:
            obstacles = obstacles0
        else:
            obstacles = None

        if wind:
            crosswind = np.array([-1.5, 0])
        else:
            crosswind = np.array([0, 0])

        t0 = 0
        t = ca.DM(t0)

        times = np.array([[0]])

        main_loop = time()  # return time in sec  

        for k in range(len(ref[:1100])):
            t1 = time()

            t = np.vstack((t,t0))
            
            if method == 'MPC':
                sol = method_cbf.get_solution(X0, u0, state_init, ref, k, obstacles)

                u = ca.reshape(sol['x'][method_cbf.n_states * (method_cbf.N + 1):], method_cbf.n_controls, method_cbf.N)
                X0 = ca.reshape(sol['x'][: method_cbf.n_states * (method_cbf.N + 1)], method_cbf.n_states, method_cbf.N+1)
                
                cat_states = np.dstack((
                    cat_states,
                    mpc.DM2Arr(X0)
                ))
                
                cat_controls = np.dstack((
                    cat_controls,
                    mpc.DM2Arr(u[:, 0])
                ))
                

                t0, state_init, u0 = mpc.shift_timestep(params.dt, t0, state_init, u, method_cbf.f, crosswind)

                X0 = ca.horzcat(
                    X0[:, 1:],
                    ca.reshape(X0[:, -1], -1, 1)
                )

                mpc_iter = mpc_iter + 1
            
            elif method == 'PID':

                cat_states = np.dstack((
                    cat_states,
                    pid.DM2Arr(state_init)
                ))

                u = method_cbf.get_u(state_init, ref, k)

                sol = method_cbf.get_solution(u, state_init, ref, k, obstacles)
                u_safe = sol['x']
                
                cat_controls = np.dstack((
                    cat_controls,
                    pid.DM2Arr(u)
                ))

                t0, state_init, u0 = pid.shift_timestep(params.dt, t0, state_init, u_safe, method_cbf.f, crosswind)

            # xx ...
            t2 = time()
            times = np.vstack((
                times,
                t2-t1
            ))
                

        main_loop_time = time()

        print('\tTotal time: ', main_loop_time - main_loop)
        print('\tavg iteration time: ', np.array(times).mean() * 1000, 'ms')

        results.append((cat_states, cat_controls))

    # simulate
    cat_states, cat_controls = results[0]
    x_target, y_target = ref[-1, 0], ref[-1, 1]
    theta_target = np.arctan2(ref[-1, 1] - ref[-2,1], ref[-1, 0] - ref[-2,0])
    
    sim_time_start = time() 
    simulate(ref, cat_states, obstacles, r, 
             t, params.dt, w, 
             np.array([x_init, y_init, theta_init, x_target, y_target, theta_target]), 
             image_path, save=True)
    
    sim_time_end = time()
    print('\tTotal simulation time: ', sim_time_end - sim_time_start)
    print('Done')