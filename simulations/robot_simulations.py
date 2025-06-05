import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
import numpy as np
from scipy import stats
import cvxpy as cp
from time import time
from casadi import *
import matplotlib.pyplot as plt


def KL_div(p, q):
    # Check if both data sets have the same shape
    if p.shape != q.shape:
        raise ValueError('Both data sets must have the same shape')

    # Add a small epsilon to avoid log(0) or division by zero
    epsilon = 1e-10  # Small value to avoid log(0) and division by zero
    p = np.clip(p, epsilon, 1)  # Clip values to avoid zero
    q = np.clip(q, epsilon, 1)  # Clip values to avoid zero

    # Normalize the binned data sets to obtain probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Calculate the KL divergence
    KL_d = np.sum(p * (np.log(p) - np.log(q)))

    return KL_d


def compute_obstacle_cost(x_axis, obs_points, n_x_1, n_x_2, obs_size):
    # Cost for the obstacles
    cost_obs_term = np.zeros(n_x_1 * n_x_2)
    var_o = (2 * obs_size) ** 2
    for o in range(obs_points.shape[1]):
        diff_vec = np.array(
            [[x_axis[0][i1] - obs_points[0, o], x_axis[1][i2] - obs_points[1, o]]
             for i1 in range(n_x_1) for i2 in range(n_x_2)]
        )
        dist_vec = np.linalg.norm(diff_vec, axis=1)
        cost_obs_term += np.exp(-0.5 / var_o * dist_vec ** 2)
    return cost_obs_term


def compute_goal_cost(x_axis, goal_points, n_x_1, n_x_2):
    # Cost for the distance from the goal
    diff_vec_g = np.array(
        [[x_axis[0][i1] - goal_points[0][0], x_axis[1][i2] - goal_points[1][0]]
         for i1 in range(n_x_1) for i2 in range(n_x_2)]
    )
    return np.linalg.norm(diff_vec_g, axis=1)


def compute_boundary_cost(x_axis, n_x_1, n_x_2, boundary_points):
    # Cost for the boundaries
    cost_b_term = np.zeros(n_x_1 * n_x_2)
    var_b = 0.01 ** 2
    for b_ind in range(2):
        dist_vec_x = np.transpose(
            np.array([[np.abs(x_axis[0][i1] - boundary_points[b_ind]) for i1 in range(n_x_1)]]
                     * n_x_2)
        ).flatten()
        dist_vec_y = np.array(
            [[np.abs(x_axis[1][i2] - boundary_points[b_ind + 2]) for i2 in range(n_x_2)]]
            * n_x_1
        ).flatten()
        cost_b_term += np.exp(-0.5 / var_b * dist_vec_x ** 2)
        cost_b_term += np.exp(-0.5 / var_b * dist_vec_y ** 2)
    return cost_b_term


def compute_cost(x_axis, obs_points, goal_points, n_x_1, n_x_2, flag_obs, obs_size, boundary_points):
    cost_obs_term = compute_obstacle_cost(x_axis, obs_points, n_x_1, n_x_2, obs_size) if flag_obs else 0
    cost_g = compute_goal_cost(x_axis, goal_points, n_x_1, n_x_2)
    cost_b_term = compute_boundary_cost(x_axis, n_x_1, n_x_2, boundary_points)
    total_cost = 150 * cost_obs_term + 0 * cost_g + 30 * cost_b_term

    return total_cost


def optimize_weights_single_step(pi_u, x_si, x_axis, u_axis, n_x_bins1, n_x_bins2, n_u_bins, tot_cost, dt, N_primitives):
    w = cp.Variable((1, N_primitives))
    constraints = [w >= 0, cp.sum(w) == 1]
    x_curr = x_si.reshape(2,)

    # Precompute grids
    x_grid = np.array([[x_axis[0][i1], x_axis[1][i2]]
                       for i1 in range(n_x_bins1) for i2 in range(n_x_bins2)])
    u_grid = np.array([[u_axis[i1], u_axis[i2]]
                       for i1 in range(n_u_bins) for i2 in range(n_u_bins)])

    # Compute q_u
    goal_u = single_integrator_position_controller(
        x_si + np.random.normal(0, 0.008, size=x_si.shape),
        goal_points[:2][:]).reshape(2,)
    q_u = stats.multivariate_normal.pdf(u_grid, goal_u, np.diag([0.005, 0.005])).reshape(n_u_bins, n_u_bins)
    q_u /= np.sum(q_u)

    KL_terms = []
    el = np.zeros((N_primitives, n_u_bins, n_u_bins))

    for z1 in range(n_u_bins):
        for z2 in range(n_u_bins):
            u_vec = np.array([u_axis[z1], u_axis[z2]])
            x_next = x_curr + u_vec * dt

            # Compute p_x and q_x
            p_x = stats.multivariate_normal.pdf(x_grid, x_next, np.diag([0.008, 0.008]))
            q_x = stats.multivariate_normal.pdf(x_grid, x_next, np.diag([0.002, 0.002]))
            p_x /= np.sum(p_x)
            q_x /= np.sum(q_x)

            # Weighted sum of primitives
            p_u_comb = sum(w[0, i] * pi_u[i, z1, z2] for i in range(N_primitives))

            # KL divergence term
            KL_terms.append(cp.sum(cp.rel_entr(p_u_comb, q_u[z1, z2])))

            # Compute future cost
            x_nn = x_next + u_vec * dt
            p_x_nn = stats.multivariate_normal.pdf(x_grid, x_nn, np.diag([0.008, 0.008]))
            p_x_nn /= np.sum(p_x_nn)

            expected_cost_x_next = np.sum(p_x_nn * tot_cost)

            expected_cost_x = np.sum(np.multiply(p_x, tot_cost + expected_cost_x_next))

            for i in range(N_primitives):
                el[i, z1, z2] = pi_u[i, z1, z2] * (expected_cost_x + KL_div(p_x, q_x))

    # Total cost
    cost = np.sum([w[0, i] * np.sum(el[i]) for i in range(N_primitives)])
    L = cp.sum(KL_terms) + cost

    # Solve
    prob = cp.Problem(cp.Minimize(L), constraints)
    result = prob.solve(solver=cp.SCS, verbose=False, max_iters=1000)

    if np.any(w.value < 0) or np.any(w.value > 1):
        w.value = np.clip(w.value, 0, 1)

    return result, w.value


def sample_policy(w, pi_u, u_axis, n_u_bins):
    p_comb = 0
    for i in range(N_primitives):
        p_comb = p_comb + w[0, i] * pi_u[i]  # Linear combination of the policies with the optimal weights
    p_comb = p_comb / np.sum(p_comb)
    # Sample from the optimal policy
    u_comb_ind = np.random.choice(n_u_bins * n_u_bins, size=1, p=p_comb.flatten(order='C'))[0]
    adjusted_index = np.unravel_index(u_comb_ind, (n_u_bins, n_u_bins), order='C')
    u_comb = np.array([u_axis[adjusted_index[0]], u_axis[adjusted_index[1]]]).reshape(2, 1)
    return u_comb


def apply_control(r, si_to_uni_dyn, u_comb, x, N):
    # Transform single integrator velocity commands to unicycle
    dxu = si_to_uni_dyn(u_comb, x)
    # Limit to avoid going out of the boundary values
    radius = r.wheel_radius
    leng = r.base_length
    dxdd = np.vstack((1 / (2 * radius) * (2 * dxu[0, :] - leng * dxu[1, :]),
                      1 / (2 * radius) * (2 * dxu[0, :] + leng * dxu[1, :])))
    to_thresh = np.absolute(dxdd) > r.max_wheel_velocity
    dxdd[to_thresh] = (r.max_wheel_velocity - 0.001) * np.sign(dxdd[to_thresh])
    dxu = np.vstack((radius / 2 * (dxdd[0, :] + dxdd[1, :]),
                     radius / leng * (dxdd[1, :] - dxdd[0, :])))

    r.set_velocities(np.arange(N), dxu)
    r.step()


def primitives(N_primitives, x_curr, n_u_bins, u_axis, boundary_points):
    # Generate 4 primitive policies using proportional controllers targeting one boundary each
    pi_u = np.zeros((N_primitives, n_u_bins, n_u_bins))

    # Proportional gain
    Kp = 2
    # Define directional goal points
    b_offsets = [
        np.array([[boundary_points[1]], x_curr[1]]),    # Right
        np.array([[boundary_points[0]], x_curr[1]]),    # Left
        np.array([x_curr[0], [boundary_points[3]]]),    # Up
        np.array([x_curr[0], [boundary_points[2]]])     # Down
    ]
    # Discrete control grid
    u_grid = np.array([[u_axis[i], u_axis[j]] for i in range(n_u_bins) for j in range(n_u_bins)])

    for i in range(N_primitives):
        error = b_offsets[i] - x_curr
        u_desired = Kp * error.flatten()
        u_desired = np.clip(u_desired, u_axis[0], u_axis[-1])

        # Generate Gaussian policy centered at u_desired
        cov = np.diag([0.005, 0.005])
        pi_u[i] = stats.multivariate_normal.pdf(u_grid, mean=u_desired, cov=cov).reshape(n_u_bins, n_u_bins)
        pi_u[i] /= np.sum(pi_u[i])

    return pi_u


max_u = 0.2     # max input
u_dim = 2       # dimension of the input
x_dim = 2       # dimension of the state
dt = 0.033      # Robotarium time-step

# Discretization points of the input
n_u_bins = 7
u_axis = np.array(np.linspace(-max_u, max_u, n_u_bins))

# Discretization points of the state
boundary_points = np.array([-1.6, 1.6, -1.0, 1.0])
n_x_bins1 = 33
n_x_bins2 = 21
x_axis = [[*np.linspace(boundary_points[0], boundary_points[1], n_x_bins1)],
          [*np.linspace(boundary_points[2], boundary_points[3], n_x_bins2)]]

nSims = 5

max_steps = 3000

is_obs = 1      # 1 if there are obstacles, 0 otherwise

plot_primitives = 1  # 1 to plot the individual primitives' trajectories, 0 otherwise

CM = ['b', 'g', 'r', 'c', 'm']  # Colors for the trajectories plot
CM_prim = ['purple', 'mediumseagreen', 'salmon', 'cornflowerblue']  # Colors for the trajectories plot

# Define goal points
goal_points = np.array(np.mat('-1.3; 0; 0'))
# Create Goal Point Markers
goal_marker_size_m = 0.1
# Text with goal identification
goal_caption = 'G'

# Define obstacles
obs_points = np.array(np.mat('0 -0.8 0.5;0.5 -0.1 -0.3'))
# Create Obstacle Points Markers
obs_marker_size_m = 0.1
obs_caption = [r'$O_{0}$'.format(ii) for ii in range(obs_points.shape[1])]

# initial_conditions = np.array(np.mat('1; 0.8; 0'))
initial_conditions = [np.array(np.mat('1.2; 0; 0')), np.array(np.mat('1.2; 0.6; 0')), np.array(np.mat('-0.3;0.8; 0')),
                      np.array(np.mat('-0.1; -0.6; 0')), np.array(np.mat('1; -0.7; 0'))]


N_primitives = 4      # Number of primitives

# Compute the cost
tot_cost = compute_cost(x_axis, obs_points, goal_points, n_x_bins1, n_x_bins2, is_obs, obs_marker_size_m, boundary_points)


# Cost Heatmap
# Create a figure and axes
fig, ax = plt.subplots(figsize=[9, 6])
plt.rcParams.update({'font.size': 14})
rotated = np.transpose(tot_cost.reshape(n_x_bins1, n_x_bins2))

# Create a heatmap using imshow()
contours = ax.contourf(x_axis[0], x_axis[1], rotated, 500, cmap='coolwarm')

# Add a colorbar
cbar = plt.colorbar(contours, shrink=0.8)
cbar.ax.tick_params(labelsize=18)

# Add labels and title
ax.set_xlabel('$p_x$ [m]', fontsize=20)
ax.set_ylabel('$p_y$ [m]', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()

plt.savefig('robot_cost_map.png', dpi=800)


N = 1   # Number of robots

# Initialize variables
position_history = np.zeros((nSims, 2, max_steps))  # trajectory of the robot
err = np.zeros((nSims, max_steps))                  # position error
w_h = np.zeros((nSims, max_steps, N_primitives))    # weights
step = np.zeros(nSims, dtype=int)                   # number of steps
count = np.zeros(nSims, dtype=int)                  # number of steps at the goal position
max_count = 60                                      # number of steps the robot has to stay at the goal position

for n_sim in range(nSims):
    # Instantiate Robotarium object
    r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions[n_sim],
                              sim_in_real_time=False)

    # Create single integrator position controller
    single_integrator_position_controller = create_si_position_controller()

    _, uni_to_si_states = create_si_to_uni_mapping()

    # Create mapping from single integrator velocity commands to unicycle velocity commands
    si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion()

    # Define x initially
    x = r.get_poses()
    x_si = uni_to_si_states(x)

    # Scale the plotted markers to be the diameter of provided argument (in meters)
    marker_size_goal = determine_marker_size(r, goal_marker_size_m)
    font_size = determine_font_size(r, 0.08)
    # Plot text for caption
    goal_points_text = [r.axes.text(goal_points[0], goal_points[1], goal_caption, fontsize=font_size, color='k',
                                    fontweight='bold', horizontalalignment='center', verticalalignment='center',
                                    zorder=-1)]
    goal_markers = [r.axes.scatter(goal_points[0], goal_points[1], s=marker_size_goal,
                                   facecolors='none', edgecolors='g', linewidth=5, zorder=-1)]

    if is_obs:
        marker_size_obs = determine_marker_size(r, obs_marker_size_m)
        obs_points_text = [
            r.axes.text(obs_points[0, ii], obs_points[1, ii], obs_caption[ii], fontsize=font_size, color='k',
                        fontweight='bold', horizontalalignment='center', verticalalignment='center',
                        zorder=-2)
            for ii in range(obs_points.shape[1])]
        obs_markers = [r.axes.scatter(obs_points[0, ii], obs_points[1, ii], s=marker_size_obs, facecolors='none',
                                      edgecolors='r', linewidth=5, zorder=-2)
                       for ii in range(obs_points.shape[1])]

    r.step()
    step[n_sim] = int(0)

    print("\nStart of Simulation", n_sim + 1, "\n")

    # While the number of robots at the required poses is less than N and step[n_sim] < max_steps:
    while step[n_sim] < max_steps and count[n_sim] < max_count:
        x = r.get_poses()
        x_si = uni_to_si_states(x)
        position_history[n_sim, :, step[n_sim]] = x_si[:2].reshape(2, )  # Save position

        # Update the marker sizes if the figure window size is changed (to be removed when submitting to the Robotarium)
        goal_markers[0].set_offsets(goal_points[:2].T)
        goal_markers[0].set_sizes([determine_marker_size(r, goal_marker_size_m)])
        if is_obs:
            for i in range(obs_points.shape[1]):
                obs_markers[i].set_offsets(obs_points[:2, i].T)
                obs_markers[i].set_sizes([determine_marker_size(r, obs_marker_size_m)])

        pi_u = primitives(N_primitives, x_si, n_u_bins, u_axis, boundary_points)

        # Optimize weights
        result, w = optimize_weights_single_step(pi_u, x_si, x_axis, u_axis, n_x_bins1, n_x_bins2, n_u_bins, tot_cost, dt, N_primitives)

        # Sample policy
        u_comb = sample_policy(w, pi_u, u_axis, n_u_bins)

        apply_control(r, si_to_uni_dyn, u_comb, x, N)

        w_h[n_sim, step[n_sim]] = w[0, :]  # save weights

        step[n_sim] = step[n_sim] + 1  # update step

        # If the robot goes out of the boundaries stop the simulation
        if (x_si[0] < boundary_points[0] or x_si[0] > boundary_points[1]
                or x_si[1] < boundary_points[2] or x_si[1] > boundary_points[3]):
            print("Robot went out of the boundaries!")
            break

        if np.size(at_pose(np.vstack((x_si, x[2, :])), goal_points, position_error=0.08, rotation_error=100)) == N:
            count[n_sim] += 1
        else:
            count[n_sim] = 0

    print("\nEnd of Simulation", n_sim + 1, "\n_______________________________________________")
    r.call_at_scripts_end()

    # Error = distance from the goal point
    err[n_sim] = np.sqrt(np.sum((position_history[n_sim, :, :] - goal_points[:2]) ** 2, axis=0))


    # Plot weights
    T = (step[n_sim] - 1) * dt
    fig, ax = plt.subplots(figsize=(9, 6))
    #plt.title('Simulation %i' % (n_sim + 1), fontsize=16)
    for i in range(N_primitives):
        ax.plot(np.linspace(0, T, step[n_sim]), w_h[n_sim, 0:step[n_sim], i], linewidth=1.8, color=CM_prim[i])
    ax.grid()
    ax.set_xlabel('time [s]', fontsize=20)
    ax.set_ylabel('$w^{(i)}$', fontsize=20)
    ax.tick_params(labelsize=18)
    #ax.set_title('Weights', fontsize=15)
    ax.set_xlim(0, T)
    ax.legend(['$w^{(1)}$', '$w^{(2)}$', '$w^{(3)}$', '$w^{(4)}$'], loc="upper right", fontsize=16)
    plt.tight_layout()
    plt.savefig('weights_simulation%i.png' % (n_sim + 1))


if plot_primitives:
    # Initialize variables
    position_history_primitives = np.zeros((N_primitives, 2, max_steps))  # trajectory of the robot
    err = np.zeros((N_primitives, max_steps))  # position error
    step_prim = np.zeros(N_primitives, dtype=int)  # number of steps

    for p in range(N_primitives):
        # Instantiate Robotarium object
        r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=np.array(np.mat('0; 0.3; 0')),
                                  sim_in_real_time=False)

        _, uni_to_si_states = create_si_to_uni_mapping()

        # Create mapping from single integrator velocity commands to unicycle velocity commands
        si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion()

        # Define x initially
        x = r.get_poses()
        x_si = uni_to_si_states(x)

        # Scale the plotted markers to be the diameter of provided argument (in meters)
        marker_size_goal = determine_marker_size(r, goal_marker_size_m)
        font_size = determine_font_size(r, 0.08)
        # Plot text for caption
        goal_points_text = [r.axes.text(goal_points[0], goal_points[1], goal_caption, fontsize=font_size, color='k',
                                        fontweight='bold', horizontalalignment='center', verticalalignment='center',
                                        zorder=-1)]
        goal_markers = [r.axes.scatter(goal_points[0], goal_points[1], s=marker_size_goal,
                                       facecolors='none', edgecolors='g', linewidth=5, zorder=-1)]

        r.step()
        step_prim[p] = int(0)

        print("\nStart of Simulation", p + 1, "\n")

        # While the number of robots at the required poses is less than N and step[p] < max_steps:
        while (np.size(at_pose(np.vstack((x_si, x[2, :])), goal_points, position_error=0.05, rotation_error=100)) != N
               and step_prim[p] < max_steps):
            x = r.get_poses()
            x_si = uni_to_si_states(x)
            position_history_primitives[p, :, step_prim[p]] = x_si[:2].reshape(2, )  # Save position

            # Update the marker sizes if the figure window size is changed (to be removed when submitting to the Robotarium)
            goal_markers[0].set_offsets(goal_points[:2].T)
            goal_markers[0].set_sizes([determine_marker_size(r, goal_marker_size_m)])
            if is_obs:
                for i in range(obs_points.shape[1]):
                    obs_markers[i].set_offsets(obs_points[:2, i].T)
                    obs_markers[i].set_sizes([determine_marker_size(r, obs_marker_size_m)])

            pi_u = primitives(N_primitives, x_si, n_u_bins, u_axis, boundary_points)

            w_vec = np.zeros((1, N_primitives))
            w_vec[0, p] = 1
            # Sample policy
            u_comb = sample_policy(w_vec, pi_u, u_axis, n_u_bins)

            apply_control(r, si_to_uni_dyn, u_comb, x, N)

            step_prim[p] = step_prim[p] + 1  # update step

            # If the robot goes out of the boundaries stop the simulation
            if (x_si[0] < boundary_points[0] or x_si[0] > boundary_points[1]
                    or x_si[1] < boundary_points[2] or x_si[1] > boundary_points[3]):
                print("Robot went out of the boundaries!")
                break

        print("\nEnd of Simulation", p + 1, "\n_______________________________________________")

        r.call_at_scripts_end()

# Plot trajectories
plt.figure()
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=np.array(np.mat('2;2;2')),
                          sim_in_real_time=True)

# Plot text for caption
goal_points_text = [r.axes.text(goal_points[0], goal_points[1], goal_caption, fontsize=font_size, color='k',
                                fontweight='bold', horizontalalignment='center', verticalalignment='center',
                                zorder=-1)]
goal_markers = [r.axes.scatter(goal_points[0], goal_points[1], s=marker_size_goal,
                               facecolors='none', edgecolors='g', linewidth=5, zorder=-1)]
if is_obs:
    # Create Obstacle Points Markers
    marker_size_obs = determine_marker_size(r, obs_marker_size_m)
    obs_caption = [r'$O_{0}$'.format(ii) for ii in range(obs_points.shape[1])]
    obs_points_text = [r.axes.text(obs_points[0, ii], obs_points[1, ii], obs_caption[ii], fontsize=font_size, color='k',
                                   fontweight='bold', horizontalalignment='center', verticalalignment='center', zorder=-2)
                       for ii in range(obs_points.shape[1])]
    obs_markers = [r.axes.scatter(obs_points[0, ii], obs_points[1, ii], s=marker_size_obs, facecolors='none',
                                  edgecolors='r', linewidth=5, zorder=-2)
                   for ii in range(obs_points.shape[1])]

for n_sim in range(nSims):
    # Plot the path
    start_m_size = determine_marker_size(r, 0.02)
    r.axes.scatter(position_history[n_sim, 0, 0], position_history[n_sim, 1, 0], s=start_m_size, linewidth=0.5,
                   color='k',
                   marker='X')
    r.axes.scatter(position_history[n_sim, 0, 1:step[n_sim]], position_history[n_sim, 1, 1:step[n_sim]], s=1,
                   linewidth=1, color=CM[n_sim], linestyle='solid')

plt.savefig('robot_trajectories.png')  # save figure with the complete trajectories

if plot_primitives:
    plt.figure()
    r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=np.array(np.mat('2;2;2')),
                              sim_in_real_time=True)

    # Plot text for caption
    goal_points_text = [r.axes.text(goal_points[0], goal_points[1], goal_caption, fontsize=font_size, color='k',
                                    fontweight='bold', horizontalalignment='center', verticalalignment='center',
                                    zorder=-1)]
    goal_markers = [r.axes.scatter(goal_points[0], goal_points[1], s=marker_size_goal,
                                   facecolors='none', edgecolors='g', linewidth=5, zorder=-1)]
    for p in range(N_primitives):
        # Plot the path of the primitives
        start_m_size = determine_marker_size(r, 0.02)
        r.axes.scatter(position_history_primitives[p, 0, 0], position_history_primitives[p, 1, 0], s=start_m_size, linewidth=0.5,
                       color='k',
                       marker='X')
        r.axes.plot(position_history_primitives[p, 0, 1:step_prim[p]], position_history_primitives[p, 1, 1:step_prim[p]],
                       linewidth=1.8, color=CM_prim[p], linestyle='solid', label=r'$\pi^{(%i)}$' % int(p+1))
    r.axes.legend(loc='upper right', bbox_to_anchor=(0.97, 0.97), fontsize=16)

    plt.savefig('robot_primitives.png')  # save figure with the complete primitives trajectories