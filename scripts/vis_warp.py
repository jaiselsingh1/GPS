import mujoco_warp as mjw 
import warp as wp 
import mujoco 
import numpy as np 
from mppi_gps.controllers import mj_warp_mppi
from mppi_gps.envs.xarm7_env import Xarm7
import time

env = Xarm7()
K = 64
T = 32 * 5 
data = mj_warp_mppi.make_data(env, K, T, njmax = 500, nconmax = 100)

init_state = np.zeros((env.model.nq + env.model.nv), dtype=np.float32)
controls = np.zeros((K, T, env.model.nu), dtype=np.float32)
d = init_state.shape[0]
rollout_states = wp.zeros((K, T+1, d), dtype=wp.float32)

start = time.time()
mj_warp_mppi.mjw_rollout(data, init_state, controls, rollout_states=rollout_states)
finish = time.time()
print(finish - start)

start = time.time()
mj_warp_mppi.mjw_rollout(data, init_state, controls, rollout_states=rollout_states)
finish = time.time()
print(finish - start)

# np_states = rollout_states.numpy()

# with wp.ScopedCapture() as capture:
#     mj_warp_mppi.mjw_rollout(data, init_state, controls, rollout_states=rollout_states)

# wp.capture_debug_dot_print(capture.graph, "./capture.dot", verbose=False)

start = time.time()
mj_warp_mppi.mjw_rollout(data, init_state, controls, rollout_states=rollout_states)
finish = time.time()
print(finish - start)
