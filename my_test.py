""" Tracks a specified trajectory on the simplified simulator using the data-augmented MPC.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""


import time
import json
import argparse
import numpy as np
from tqdm import tqdm
from src.utils.utils import separate_variables
from src.utils.quad_3d_opt_utils import get_reference_chunk
from src.utils.trajectories import loop_trajectory, lemniscate_trajectory, check_trajectory, minimum_snap_trajectory_generator
from src.utils.visualization import initialize_drone_plotter, draw_drone_simulation, trajectory_tracking_results
from src.experiments.comparative_experiment import prepare_quadrotor_mpc
from config.configuration_parameters import SimpleSimConfig

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.utils.keyframe_3d_gen import center_and_scale
from src.utils.trajectory_generator import draw_poly, get_full_traj, fit_multi_segment_polynomial_trajectory

def main(args):

    params = {
        "version": args.model_version,
        "name": args.model_name,
        "reg_type": args.model_type,
        "quad_name": "my_quad"
    }
    simulation_options = SimpleSimConfig.simulation_disturbances
    quad_mpc = prepare_quadrotor_mpc(simulation_options, **params)
    quad = quad_mpc.quad
    n_mpc_nodes = quad_mpc.n_nodes
    t_horizon = quad_mpc.t_horizon
    simulation_dt = quad_mpc.simulation_dt
    reference_over_sampling = 5
    control_period = t_horizon / (n_mpc_nodes * reference_over_sampling)



    pos_traj, att_traj = generate_waypoints()
    #pos_traj, att_traj = point2trajectory()
    # 确保没有负数z出现
    min_z_pos = np.min(pos_traj[:, -1])
    if min_z_pos < 0:
        pos_traj[:, -1] = pos_traj[:, -1] + 2 * min_z_pos * np.sign(min_z_pos)
    # 基于两个轨迹点位置之间的平均距离来计算可行的时间
    av_dist = np.mean(np.sqrt(np.sum(np.diff(pos_traj, axis=0) ** 2, axis=1)))
    av_dt = av_dist / args.speed
    print(av_dt)
    #计算位置轨迹的多项式拟合，并计算完整的运动学参考。这个轨迹是根据 MPC 所需的频率进行采样的。
    #print(np.shape(pos_traj))=(10,3)
    poly_pos_traj = fit_multi_segment_polynomial_trajectory(pos_traj.T, att_traj[:, -1].T)
    discretization_dt=0.1
    traj, yaw, t_ref = get_full_traj(poly_pos_traj, av_dt, discretization_dt)
    reference_traj, t_ref, reference_u = minimum_snap_trajectory_generator(traj, yaw, t_ref, quad, None, False)
    draw_poly(reference_traj, reference_u, t_ref, pos_traj.T)
    #下面代码仅试验




def point2trajectory():
    #现在是高斯过程随变搞的，没有输入，还需修改，理想情况：输入很多（100）个点，高斯过程插值
    random_state = np.random.randint(0, 9999)
    print(random_state)
    kernel_z = ExpSineSquared(length_scale=5.5, periodicity=60)
    kernel_y = ExpSineSquared(length_scale=4.5, periodicity=30) + ExpSineSquared(length_scale=4.0, periodicity=15)
    kernel_x = ExpSineSquared(length_scale=4.5, periodicity=30) + ExpSineSquared(length_scale=4.5, periodicity=60)

    gp_x = GaussianProcessRegressor(kernel=kernel_x)
    gp_y = GaussianProcessRegressor(kernel=kernel_y)
    gp_z = GaussianProcessRegressor(kernel=kernel_z)

    # High resolution sampling for track boundaries
    inputs_x = np.linspace(0, 60, 100)
    inputs_y = np.linspace(0, 30, 100)
    inputs_z = np.linspace(0, 60, 100)

    x_sample_hr = gp_x.sample_y(inputs_x[:, np.newaxis], 1, random_state=random_state)
    y_sample_hr = gp_y.sample_y(inputs_y[:, np.newaxis], 1, random_state=random_state)
    z_sample_hr = gp_z.sample_y(inputs_z[:, np.newaxis], 1, random_state=random_state)
    print(len(x_sample_hr))
    print(x_sample_hr[0])

    max_x_coords = np.max(x_sample_hr, 0)
    max_y_coords = np.max(y_sample_hr, 0)
    max_z_coords = np.max(z_sample_hr, 0)
    print(max_x_coords)


    min_x_coords = np.min(x_sample_hr, 0)
    min_y_coords = np.min(y_sample_hr, 0)
    min_z_coords = np.min(z_sample_hr, 0)
    print(min_x_coords)

    x_sample_hr, y_sample_hr, z_sample_hr = center_and_scale(
        x_sample_hr, y_sample_hr, z_sample_hr,
        max_x_coords, min_x_coords, max_y_coords, min_y_coords, max_z_coords, min_z_coords)
    print(x_sample_hr[0][0])

    # Low resolution for control points
    lr_ind = np.linspace(0, len(x_sample_hr) - 1, 10, dtype=int)
    lr_ind[-1] = 0
    x_sample_lr = x_sample_hr[lr_ind, :]
    y_sample_lr = y_sample_hr[lr_ind, :]
    z_sample_lr = z_sample_hr[lr_ind, :]

    x_sample_diff = x_sample_hr[lr_ind + 1, :] - x_sample_lr
    y_sample_diff = y_sample_hr[lr_ind + 1, :] - y_sample_lr
    z_sample_diff = z_sample_hr[lr_ind + 1, :] - z_sample_lr

    # Get angles at keypoints
    a_x = np.arctan2(z_sample_diff, y_sample_diff) 
    a_y = np.arctan2(x_sample_diff, z_sample_diff) 
    a_z = (np.arctan2(y_sample_diff, x_sample_diff) - np.pi/4)
    

    plot = False
    if plot:
        # Plot checking
        # Build rotation matrices
        rx = [np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]) for a in a_x[:, 0]]
        ry = [np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]) for a in a_y[:, 0]]
        rz = [np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]) for a in a_z[:, 0]]

        main_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        quiver_axes = np.zeros((len(lr_ind), 3, 3))

        for i in range(len(lr_ind)):
            r_mat = rz[i].dot(ry[i].dot(rx[i]))
            rot_body = r_mat.dot(main_axes)
            quiver_axes[i, :, :] = rot_body

        shortest_axis = min(np.max(x_sample_hr) - np.min(x_sample_hr),
                            np.max(y_sample_hr) - np.min(y_sample_hr),
                            np.max(z_sample_hr) - np.min(z_sample_hr))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_sample_lr, y_sample_lr, z_sample_lr)
        ax.plot(x_sample_hr[:, 0], y_sample_hr[:, 0], z_sample_hr[:, 0], '-', alpha=0.5)
        ax.quiver(x_sample_lr[:, 0], y_sample_lr[:, 0], z_sample_lr[:, 0],
                  x_sample_diff[:, 0], y_sample_diff[:, 0], z_sample_diff[:, 0], color='g',
                  length=shortest_axis / 6, normalize=True, label='traj. norm')
        ax.quiver(x_sample_lr, y_sample_lr, z_sample_lr,
                  quiver_axes[:, 0, :], quiver_axes[:, 1, :], quiver_axes[:, 2, :], color='b',
                  length=shortest_axis / 6, normalize=True, label='quad. att.')
        ax.tick_params(labelsize=16)
        ax.legend(fontsize=16)
        ax.set_xlabel('x [m]', size=16, labelpad=10)
        ax.set_ylabel('y [m]', size=16, labelpad=10)
        ax.set_zlabel('z [m]', size=16, labelpad=10)
        ax.set_xlim([np.min(x_sample_hr), np.max(x_sample_hr)])
        ax.set_ylim([np.min(y_sample_hr), np.max(y_sample_hr)])
        ax.set_zlim([np.min(z_sample_hr), np.max(z_sample_hr)])
        ax.set_title('Source keypoints', size=18)
        plt.show()
    #画完生成轨迹
    curve = np.concatenate((x_sample_lr, y_sample_lr, z_sample_lr), 1)
    attitude = np.concatenate((a_x, a_y, a_z), 1)
    print("xiba")
    print(np.shape(curve))
    print(np.shape(attitude))
    #print("curve 的第一个元素是：{}，长度是：{}".format(curve[0], len(curve)))
    #print("attitude 的第一个元素是：{}，长度是：{}".format(attitude[0], len(attitude))) 

    return curve, attitude   

def generate_waypoints():
    # 生成时间向量
    t = np.linspace(0, 2*np.pi, 100)

    # 生成S型轨迹的位置
    x = t
    y = np.sin(t)
    z = np.cos(t)
    

    # 计算机器人在路径点的朝向
    dx = np.gradient(x)
    dy = np.gradient(y)
    dz = np.gradient(z)
    norm = np.sqrt(dx**2 + dy**2 + dz**2)

    # 归一化
    dx /= norm
    dy /= norm
    dz /= norm
    

    # 将x, y, z和dx, dy, dz堆叠到一个新的numpy数组中
    pos = np.stack((x, y, z), axis=-1)
    dir = np.stack((dx, dy, dz), axis=-1)
    print(np.shape(pos))
    print(np.shape(dir))

    # 绘制S型轨迹和机器人的朝向
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='S型轨迹')
    ax.quiver(x[::10], y[::10], z[::10], dx[::10], dy[::10], dz[::10], color='r', label='机器人朝向')
    ax.legend()
    plt.show()

    return pos, dir



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_version", type=str, default="",
                        help="Version to load for the regression models. By default it is an 8 digit git hash.")

    parser.add_argument("--model_name", type=str, default="",
                        help="Name of the regression model within the specified <model_version> folder.")

    parser.add_argument("--model_type", type=str, default="gp", choices=["gp", "rdrv"],
                        help="Type of regression model (GP or RDRv linear)")

    parser.add_argument("--trajectory", type=str, default="loop", choices=["loop", "lemniscate"],
                        help='path to other necessary data files (eg. vocabularies)')

    parser.add_argument("--max_speed", type=float, default=8,
                        help="Maximum axial speed executed during the flight in m/s. For the `loop` trajectory, "
                             "velocities are feasible up to 14 m/s, and for the `lemniscate` up to 8 m/s")

    parser.add_argument("--acceleration", type=float, default=1,
                        help="Acceleration of the reference trajectory. Higher accelerations will shorten the execution"
                             "time of the tracking")

    parser.add_argument("--trajectory_radius", type=float, default=5, help="Radius of the reference trajectories")
    parser.add_argument("--speed", type=float, default="1")
    input_arguments = parser.parse_args()

    main(input_arguments)
