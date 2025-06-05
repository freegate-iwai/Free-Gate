This folder contains the code for the experiments in the simulated Robotarium environment.
The optimal weights are computed by solving the minimization problem illustrated in the paper at each time step $k$:

```math
\begin{split}
    \min_{\mathbf{w}_{k}\in \Delta^{n_{\pi}}} \quad & {D}_{KL} \left(p_{k}\left(\mathbf{x}_{k},\mathbf{u}_{k}\mid \mathbf{x}_{{k-1}}\right) \,\big\|\, q_{k}\left(\mathbf{x}_{k},\mathbf{u}_{k}\mid \mathbf{x}_{k-1} \right) \right) + \mathbb{E}_{p_{k}\left(\mathbf{x}_{k},\mathbf{u}_{k}\mid \mathbf{x}_{k-1}\right)}\left[c_{k}^{\textup{x}}\left(\mathbf{X}_{k}\right) + c_{k}^{\textup{u}}\left(\mathbf{U}_{k}\right) + l_{k}^{\star}\left(\mathbf{X}_{k-1}\right) \right]\\
    \mbox{s.t.} & \ p_{\mathbf{u},{k}}\left(\mathbf{u}_{{k}}\mid \mathbf{x}_{{k-1}} \right)  = \sum_{i\in \mathcal{P}}\mathbf{w}_{k}^{(i)} \pi^{({i})}\left(\mathbf{u}_{k}\mid \mathbf{x}_{k-1} \right).
\end{split}
```

In order to improve the computational efficiency, the cost function is rewritten, by exploiting the definitions and substituting the constraint $` p_{\mathbf{u},{k}}\left(\mathbf{u}_{{k}}\mid \mathbf{x}_{{k-1}} \right)  = \sum_{i\in \mathcal{P}}\mathbf{w}_{k}^{(i)} \pi^{({i})}\left(\mathbf{u}_{k}\mid \mathbf{x}_{k-1} \right) `$, as follows:

```math
{D}_{KL} \left( p_{\mathbf{u},{k}}\left(\mathbf{u}_{{k}}\mid \mathbf{x}_{{k-1}} \right)   \,\big\|\,  q_{\mathbf{u},{k}}\left(\mathbf{u}_{{k}}\mid \mathbf{x}_{{k-1}} \right)  \right) +
\sum_{i \in \mathcal{P}} \mathbf{w}_{k}^{(i)} \mathbb{E}_{\pi^{({i})}\left(\mathbf{u}_{{k}}\mid \mathbf{x}_{{k-1}} \right)} \left[
    {D}_{KL} \left(p_{\mathbf{x},{k}} \left(\mathbf{x}_{{k}}\mid \mathbf{x}_{{k-1}}, \mathbf{u}_{{k}} \right)  \,\big\|\,   q_{\mathbf{x},{k}} \left(\mathbf{x}_{{k}}\mid \mathbf{x}_{{k-1}}, \mathbf{u}_{{k}} \right) \right)
    + c_{k}^{\textup{u}}\left(\mathbf{U}_{k}\right)
    + \mathbb{E}_{p_{\mathbf{x},{k}} \left(\mathbf{x}_{{k}} \mid \mathbf{x}_{{k-1}}, \mathbf{u}_{{k}} \right)} \left[ c_{k}^{\textup{x}}\left(\mathbf{X}_{k}\right) + l_{k}^{\star}\left(\mathbf{X}_{k-1}\right) \right]
\right].
```

We construct $`n_{\pi}=4`$ primitives, each targeting a direction by moving toward the corresponding boundary.
Specifically, the primitives are constructed as discretized Gaussians of the form $`\pi^{({i})}\left(\mathbf{u}_{k}\mid \mathbf{x}_{k-1} \right) = \mathcal{N}(\bar{\mathbf{u}}_k^{(i)}, \Sigma_u)`$. For these Gaussians, we set $`\Sigma_u= 0.005 \mathbf{I}_2`$, while $`\bar{\mathbf{u}}_k^{(i)}`$ is a vector signal (e.g., coming from a simple proportional controller) proportional to the distance between the robot position and each of the boundaries of the Robotarium work area:

```math
\bar{\mathbf{u}}_k^{(1)} = k_p \cdot \begin{bmatrix} 1.6 - p_{x,k-1} \\ 0 \end{bmatrix}, \ \bar{\mathbf{u}}_k^{(2)} = k_p \cdot \begin{bmatrix} -1.6 - p_{x,k-1} \\ 0 \end{bmatrix}, \ \bar{\mathbf{u}}_k^{(3)} = k_p \cdot \begin{bmatrix} 0 \\ 1.0 - p_{y,k-1} \end{bmatrix}, \ \bar{\mathbf{u}}_k^{(4)} = \begin{bmatrix} 0 \\ -1.0 - p_{y,k-1} \end{bmatrix},
```

where we set the proportional gain to $k_p=2$.
These primitives allow the robot to move along four cardinal directions (right, left, up, down, respectively).


In addition, to equip the agent with planning abilities without implementing the full backward recursion in the algorithm, we use a one-step rollout heuristic, recomputed at each time step, to approximate the cost-to-go as:

```math
\tilde{l}_{k+1}\left(\mathbf{X}_{k}\right)=\mathbb{E}_{\tilde{p}_{{\mathbf{x}},{k+1}} \left(\mathbf{x}_{{k+1}}{\mid} \mathbf{x}_{{k}}, \mathbf{u}_{{k+1}} \right)}[\tilde{c}_{k+1}^{\textup{x}}\left(\mathbf{X}_{k+1}\right)],
```

where we set $`\tilde{c}_{k+1}^{\textup{x}}\left(\mathbf{X}_{k+1}\right)=c_{k}^{\textup{x}}\left(\mathbf{X}_{k}\right)`$ and $`\tilde{p}_{{\mathbf{x}},{k+1}} \left(\mathbf{x}_{{k+1}}{\mid} \mathbf{x}_{{k}}, \mathbf{u}_{{k+1}} \right)=\mathcal{N}(\mathbf{x}_{k-1} + 2\mathbf{u}_k dt, \Sigma_{x})`$.
This choice is equivalent to estimating the expectation of the cost at the next time step assuming that the input $`\mathbf{u}_k`$ is applied at time step $k+1$ too.

The solution is found by using [CVXPY](https://www.cvxpy.org) with [SCS](https://www.cvxgrp.org/scs/).

The code produces the following plots:
  - The heat map of the cost $`c_{k}^{\textup{x}}\left(\mathbf{x}_{k}\right)`$
   
    ![robot_cost_map](https://github.com/user-attachments/assets/74c09293-6e9f-4664-84ec-036422e1818f)

    
  - The trajectories of the robot from 5 different initial conditions
    
    ![robot_trajectories](https://github.com/user-attachments/assets/20d5d5b1-3bc0-41b8-8ef3-eed9e28faed3)


  - The optimal weights over time for each simulation

    ![weights_simulation1](https://github.com/user-attachments/assets/dc06aa98-0d18-4585-a8d6-62a686d66b88)

    
  - The trajectories of the robot controlled by the individual primitives

    ![robot_primitives](https://github.com/user-attachments/assets/68556f12-93a3-437c-b084-e95dae894278)
