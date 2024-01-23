#直接放在src/experiments下
import time
import json
import argparse
import numpy as np
from tqdm import tqdm
from src.utils.utils import separate_variables
from src.utils.quad_3d_opt_utils import get_reference_chunk
from src.utils.trajectories import loop_trajectory, lemniscate_trajectory, check_trajectory
from src.utils.visualization import initialize_drone_plotter, draw_drone_simulation, trajectory_tracking_results
from src.experiments.comparative_experiment import prepare_quadrotor_mpc
from config.configuration_parameters import SimpleSimConfig

#全局变量
quad_current_state = None
target_point = None

def main():
	rospy.init_node("mpc")
	params = {
        "version": None,
        "name": None,
        "reg_type": None,
        "quad_name": "my_quad"
    }
	simulation_options = SimpleSimConfig.simulation_disturbances#四项干扰设置
	quad_mpc = prepare_quadrotor_mpc(simulation_options, **params)

	my_quad = quad_mpc.quad
    n_mpc_nodes = quad_mpc.n_nodes
    t_horizon = quad_mpc.t_horizon
    simulation_dt = quad_mpc.simulation_dt
    reference_over_sampling = 5
    control_period = t_horizon / (n_mpc_nodes * reference_over_sampling)

	#输入自身状态信息，储存在quad_current_state中
	rospy.Subscriber("/Vicon", String, statecallback)
	#输入目标信息，储存在target_point中
	rospy.Subscriber("/Target", String, targetcallback)
	my_quad.set_state(quad_current_state)

	#设置初始输入，实际上没有参考u
	ref_u = reference_u[0, :]

	#创建Nx状态数量矩阵，用来存储UAV在每个时间戳的预测状态
    quad_trajectory = np.zeros((len(reference_timestamps), len(quad_current_state)))

    #创建Nx4矩阵，用来存储优化后的控制输入序列
    u_optimized_seq = np.zeros((len(reference_timestamps), 4))

    # Sliding reference trajectory initial index
    current_idx = 0

    # Measure the MPC optimization time
    mean_opt_time = 0.0

    # Measure total simulation time
    total_sim_time = 0.0
	for current_idx in n_mpc_nodes:

	    #把当前状态加入第一行，参考输入加入当前
	    quad_trajectory[current_idx, :] = np.expand_dims(quad_current_state, axis=0)
	    u_optimized_seq[current_idx, :] = np.reshape(ref_u, (1, -1))

	    #

		#设置OCP的参考值
		model_ind = quad_mpc.quad_opt.set_reference_state(x_reference=target_point, u_reference=None)
	    
	    #开算
		t_opt_init = time.time()
        w_opt, x_pred = quad_mpc.optimize(use_model=model_ind, return_x=True)
        mean_opt_time += time.time() - t_opt_init

        ref_u = np.squeeze(np.array(w_opt[:4]))


		#输入参考路径，ReferenceTrajectory指我们自定义的数据类型
		#ref_sub = rospy.Subscriber("/reference_topic", ReferenceTrajectory, self.reference_callback)

	#输出控制指令
	pub = rospy.Publisher('/Cmd', String, queue_size=10)
	rate = rospy.Rate(10)
	rate.sleep()

def statecallback(data):
    quad_current_state = [data.pos, data.angle, data.vel, data.a_rate]

def targetcallback(data):
    target_point = [data.pos.x, data.pos.y, data.pos.z]

if __name__ == "__main__":
    main()

