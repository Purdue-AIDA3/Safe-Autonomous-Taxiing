import numpy as np 
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import animation
from time import time


def simulate(ref, cat_states, obstacles, r, 
             t, step_horizon, runway_w, 
             reference, image_path, save=False):
    def create_triangle(state=[0,0,0], h=40, w=20, update=False):
        x, y, th = state
        triangle = np.array([
            [h, 0   ],
            [0,  w/2],
            [0, -w/2],
            [h, 0   ]
        ]).T
        rotation_matrix = np.array([
            [cos(th), -sin(th)],
            [sin(th),  cos(th)]
        ])

        coords = np.array([[x, y]]) + (rotation_matrix @ triangle).T
        if update == True:
            return coords
        else:
            return coords[:3, :]

    def init():
        return path, current_state, target_state,

    def animate(i):
        # get variables
        x = cat_states[0, 0, i]
        y = cat_states[1, 0, i]
        th = cat_states[2, 0, i]

        # update path
        if i == 0:
            path.set_data(np.array([]), np.array([]))
        x_new = np.hstack((path.get_xdata(), x))
        y_new = np.hstack((path.get_ydata(), y))
        path.set_data(x_new, y_new)

        # update horizon
        x_new = cat_states[0, :, i]
        y_new = cat_states[1, :, i]

        # update current_state
        current_state.set_xy(create_triangle([x, y, th], update=True))

        return path, current_state, target_state,

    image_raw = mpimg.imread(image_path)  # Load the airport map image
    image = np.flipud(image_raw)
    
    # create figure and axes
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.tight_layout()

    ax.set_aspect('equal')

    ax.plot(ref[:,0], ref[:,1], 'r--', linewidth=2)
    for i in range(len(ref)):
        ax.add_patch(plt.Circle((ref[i,0], ref[i,1]), runway_w, color='g', alpha=0.01))

    for obs in obstacles:
        ax.add_patch(plt.Circle(obs, r, color='r', alpha=0.5))
    
    plt.imshow(image)

    # plt.tight_layout()

    # create lines:
    #   path
    path, = ax.plot([], [], 'k', linewidth=2)
    #   horizon
    horizon, = ax.plot([], [], 'x-g', alpha=0.5)
    #   current_state
    current_triangle = create_triangle(reference[:3])
    current_state = ax.fill(current_triangle[:, 0], current_triangle[:, 1], color='c',zorder=10)
    current_state = current_state[0]
    #   target_state
    target_triangle = create_triangle(reference[3:])
    target_state = ax.fill(target_triangle[:, 0], target_triangle[:, 1], color='b',zorder=10)
    target_state = target_state[0]

    sim = animation.FuncAnimation(
        fig=fig,
        func=animate,
        init_func=init,
        frames=len(t),
        interval=step_horizon*100,
        blit=True,
        repeat=True
    )

    if save == True:
        sim.save('./animations/animation' + str(time()) +'.gif', writer='ffmpeg', fps=30)

    return