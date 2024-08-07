"""
qfly | Qualisys Drone SDK Example Script: Multi Crazyflie 
Drones take off and fly circles around Z axis.

ESC to land at any time.
"""

import numpy as np
import pynput
import argparse

from time import sleep, time
from functools import partial

from qfly import Pose, QualisysCrazyflie, World, ParallelContexts

from src.animation_code import simulate
from src.utils import gen_waypoints, gen_reference_trajectory
from src.taxiway_graph import TaxiwayGraph

def get_taxi_ref(graph, start, end, args):
    shortest_path_flight, control_points = graph.find_shortest_path_flight(start, end)

    waypoints = gen_waypoints(control_points, args.turn_radius, args.turn_resolution)

    ref = gen_reference_trajectory(waypoints, args.v_target, args.dt)
    
    return ref

# Watch key presses with a global variable
last_key_pressed = None

# Set up keyboard callback
def on_press(key):
    """React to keyboard."""
    global last_key_pressed
    last_key_pressed = key

# if __name__ == "__main__":
parser = argparse.ArgumentParser()
parser.add_argument('--dt', type=float, default=0.1)
parser.add_argument('--v-target', type=float, default=0.2)
parser.add_argument('--z-target', type=float, default=0.3)
parser.add_argument('--turn-radius', type=float, default=0.05)
parser.add_argument('--turn-resolution', type=int, default=10)
parser.add_argument('--speed-limit', type=float, default=1)

args = parser.parse_args()

# SETTINGS
# QTM rigid body names

cf_body_names = ['cf1']  # QTM rigid body name
# cf_body_names = ['cf1', 'cf2']  # QTM rigid body name
# cf_body_names = ['cf1', 'cf3', 'cf4']  # QTM rigid body name
# cf_body_names = ['cf1', 'cf3', 'cf5']  # QTM rigid body name

cf_uris = [
           'radio://0/80/1M/E7E7E7E7E1',
        #    'radio://0/80/1M/E7E7E7E7E2',
        #    'radio://0/80/1M/E7E7E7E7E3',
        #    'radio://1/80/2M/E7E7E7E7E4',
        #    'radio://1/80/2M/E7E7E7E7E5',
        #    'radio://1/80/2M/E7E7E7E7E6'
           ]  # Crazyflie address

# led_id_base = [1,3,4,2]
# cf_marker_ids = [[i*4 + j for j in led_id_base] for i in range(len(cf_uris))]

cf_marker_ids = [
                 [31, 33, 34, 32],
                #  [5, 7, 8, 6],
                #  [9, 11, 12, 10],
                #  [13, 15, 16, 14],
                #  [17, 19, 20, 18],
                #  [21, 23, 24, 22]
                 ]

# Listen to the keyboard
listener = pynput.keyboard.Listener(on_press=on_press)
listener.start()

# Set up world - the World object comes with sane defaults
world = World(expanse=15)

# Stack up context managers
_qcfs = [QualisysCrazyflie(cf_body_name, cf_uri, world, marker_ids=cf_marker_id, qtm_ip='192.168.123.2') 
            for cf_body_name, cf_uri, cf_marker_id in zip(cf_body_names, cf_uris, cf_marker_ids)]

#################################################################
csv_file_path = r".\data\Position_of_Airport_PURT.csv"

# Create the TaxiwayGraph object
taxiway_graph = TaxiwayGraph(csv_file_path)

# Define reference signal for taxiing to takeoff
start1, end1= 'G1', 'RW1'
ref1 = get_taxi_ref(taxiway_graph, start1, end1, args)

# Define reference signal for taxiing after landing
start2, end2 = 'RW2', 'G2'
ref2 = get_taxi_ref(taxiway_graph, start2, end2, args)

mission = np.array([
                    np.array([ref1[-1,0] , ref1[-1,1]]),
                    np.array([-5.652 , 0.460]),# Take off along runway 1 [NEED TO COLLECT]
                    np.array([-1.45 , 1.316]), # visit point 3
                    np.array([ 0.31 ,  3.875]), # visit point 1
                    # np.array([1.632 , 1.291]), # visit point 2
                    np.array([ -5.574, -0.292]),# Go to end of runway 2 [NEED TO COLLECT]
                    np.array([ref2[0, 0], ref2[0, 1]]) # Land along runway 2
                    ])
ref3 = gen_reference_trajectory(mission, v=.7, dt=args.dt)

k1, k2, m = 0, 0, 0
#################################################################
last_state = None
TAKEOFF, TAXI1, MISSON, TAXI2 = True, True, True, True 
with ParallelContexts(*_qcfs) as qcfs:
    # for qcf in qcfs:
    #     qcf.set_speed_limit(args.speed_limit)

    # Let there be time
    t = time()
    dt = 0

    print("Beginning maneuvers...")
    # "fly" variable used for landing on demand
    fly = True

    # MAIN LOOP WITH SAFETY CHECK
    while(fly and all(qcf.is_safe() for qcf in qcfs)):
        # Land with Esc
        if last_key_pressed == pynput.keyboard.Key.esc:
            break

        # Mind the clock
        dt = time() - t
        t1 = time()
        states = []
        # print(last_state)
        if last_state:
            for i, qcf in enumerate(qcfs):
                states.append(np.array([float(qcf.pose.x),  float(qcf.pose.y), float((float(qcf.pose.x) - float(last_state[i][0]))/(t2-t1)), float((float(qcf.pose.y) - float(last_state[i][1]))/(t2-t1))]))
        else:
            for qcf in qcfs:
                states.append(np.array([float(qcf.pose.x),  float(qcf.pose.y), 0, 0]))
        t2 = time()
                
        last_state = states
        # print('states', states)
        if dt < 2:
            # Take off in place
            if TAKEOFF:
                print('Starting')
                TAKEOFF = False
            for idx, qcf in enumerate(qcfs):
                target = Pose(states[idx][0], states[idx][1], args.z_target)
                # Engage
                qcf.safe_position_setpoint(target)
                sleep(0.1)

        elif k1 < len(ref1):
            if TAXI1:
                print('Taxiing to take off')
                TAXI1 = False
            # Cycle all drones
            for idx, qcf in enumerate(qcfs):
                target = Pose(ref1[k1,0], ref1[k1,1], args.z_target)
                # Engage
                qcf.safe_position_setpoint(target)
                sleep(args.dt)
                k1 += 1
        
        elif k1 >= len(ref1) and m < len(ref3):
            if MISSON:
                print('Executing mission')
                MISSON = False
            for idx, qcf in enumerate(qcfs):
                target = Pose(ref3[m,0], ref3[m,1], 1.5)
                # Engage
                qcf.safe_position_setpoint(target)
                
                sleep(args.dt)
                m += 1

        elif k1 >= len(ref1) and m >= len(ref3) and k2 < len(ref2):
            if TAXI2:
                print('Taxiing from landing')
                TAXI2 = False

            for idx, qcf in enumerate(qcfs):
                target = Pose(ref2[k2,0], ref2[k2,1], args.z_target)
                # Engage
                qcf.safe_position_setpoint(target)
                sleep(args.dt)
                k2 += 1
        
        elif k2 >= len(ref2):
            print('Finishing')
            fly = False

    # Land
    while (qcf.pose.z > 0.1):
        for idx, qcf in enumerate(qcfs):
            qcf.land_in_place()
            sleep(0.02)
    