mass = 1
I0 = .2

# setting matrix_weights' variables
Q_x = 1000
Q_y = 1000
Q_theta = 1000
Q_vx = 1000
Q_vy = 1000
Q_omega = 1000

R1 = 1000
R2 = 1000

dt = .1   # time between steps in seconds
N = 60              # number of look ahead steps

v_target = 3    # m/s

F_max = 2       # Netwons
F_min = -2

tau_max = 10    # Torque
tau_min = -10

vx_min = -10
vy_min = -10
vx_max = 10
vy_max = 10

omega_min = -.5
omega_max = .5