import mujoco_warp as mjw 
import warp as wp 
import mujoco 
import numpy as np 
from mppi_gps.controllers import mj_warp_mppi
from mppi_gps.envs.xarm7_env import Xarm7


env = Xarm7()
K = 2
T = 1
data = mj_warp_mppi.make_data(env, K, T, njmax = 121)

init_state = np.zeros((env.model.nq + env.model.nv), dtype=np.float32)
controls = np.zeros((K, T, env.model.nu), dtype=np.float32)

rollout_states = mj_warp_mppi.mjw_rollout(data, init_state, controls)

np_states = rollout_states.numpy()
print(np_states[1, 1])
print("new state")
print(np_states[0, 1])