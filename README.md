# Free-Gate: Planning, Control And Policy Composition Via Free Energy Gating

This repository contains the Python code developed for the paper *Free-Gate: Planning, Control And Policy Composition Via Free Energy Gating*.

The application involves the navigation of a unicycle robot from the Robotarium [[1]](#1) in an environment with obstacles. Specifically:
* The folder *simulations* contains the code for the simulated environment. The outputs are:
  - The heat map of the cost used in the simulations:

    ![robot_cost_map](https://github.com/user-attachments/assets/e50863a9-a236-4b0c-95dd-d904ba398e22)



  - The trajectories of the robot controlled by Free-Gate:
 
    ![robot_trajectories](https://github.com/user-attachments/assets/f4554baf-fa76-4dca-b2d1-829b9053be4a)

 

  - The optimal weights computed by Free-Gate for each of the simulations:
 
    ![weights_simulation1](https://github.com/user-attachments/assets/ee0dd040-4f19-4c7e-a4b6-1f6e6310b8e5)



  - The trajectories of the robot controlled by the individual primitives:

    ![robot_primitives](https://github.com/user-attachments/assets/38ec9155-64b7-4818-a2be-c695726441d3)



* The folder *real_experiment* contains the code developed for the real-hardware experiment, which can be reproduced by uploading the file on the Robotarium webpage. The resulting video is shown here:





The Python simulator for Robotarium is available at https://github.com/robotarium/robotarium_python_simulator.

The following dependencies are required for the simulations:
- NumPy (http://www.numpy.org)
- matplotlib (http://matplotlib.org/index.html)
- CVXPY (https://www.cvxpy.org)
- SciPy (https://scipy.org).




## References
<a id="1">[1]</a> 
S. Wilson et al., "The Robotarium: Globally Impactful Opportunities, Challenges, and Lessons Learned in Remote-Access, Distributed Control of Multirobot Systems," in IEEE Control Systems Magazine, vol. 40, no. 1, pp. 26-44, Feb. 2020
