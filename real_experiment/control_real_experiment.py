import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
import numpy as np
from scipy import stats
import cvxpy as cp
from time import time


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
    result = prob.solve(solver=cp.SCS, verbose=False, max_iters=1000)#, warm_start=True)

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


is_obs = 1          # 0 if not considering obstacles, 1 otherwise


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


N = 1   # Number of robots
# Instantiate Robotarium object
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=np.array(np.mat('1.2; 0; 0')),
                          sim_in_real_time=True)


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
    obs_points_text = [r.axes.text(obs_points[0, ii], obs_points[1, ii], obs_caption[ii], fontsize=font_size, color='k',
                                   fontweight='bold', horizontalalignment='center', verticalalignment='center',
                                   zorder=-2)
                       for ii in range(obs_points.shape[1])]
    obs_markers = [r.axes.scatter(obs_points[0, ii], obs_points[1, ii], s=marker_size_obs, facecolors='none',
                                  edgecolors='r', linewidth=5, zorder=-2)
                   for ii in range(obs_points.shape[1])]


N_primitives = 4      # Number of primitives

# Compute the cost
tot_cost = compute_cost(x_axis, obs_points, goal_points, n_x_bins1, n_x_bins2, is_obs, obs_marker_size_m, boundary_points)


# Simulation
# Initialize variables
nSims = 1           # Number of simulations
max_steps = 3000    # max number of steps per simulation
r.step()
step = int(0)       # Initialize number of steps
count = 0                  # number of steps at the goal position
max_count = 60

# While the number of robots at the required poses is less than N and step < max_steps:
while step < max_steps and count < max_count:
    # Get poses of agents
    x = r.get_poses()
    x_si = uni_to_si_states(x)

    pi_u = primitives(N_primitives, x_si, n_u_bins, u_axis, boundary_points)

    # Optimize weights
    result, w = optimize_weights_single_step(pi_u, x_si, x_axis, u_axis, n_x_bins1, n_x_bins2, n_u_bins, tot_cost, dt,
                                             N_primitives)

    # Sample policy
    u_comb = sample_policy(w, pi_u, u_axis, n_u_bins)

    apply_control(r, si_to_uni_dyn, u_comb, x, N)

    step = step + 1

    if np.size(at_pose(np.vstack((x_si, x[2, :])), goal_points, position_error=0.08, rotation_error=100)) == N:
        count += 1
    else:
        count = 0


r.call_at_scripts_end()
