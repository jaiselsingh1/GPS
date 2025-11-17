import mujoco 
from jaxtyping import Float
import numpy as np
from mppi_gps.controllers.mppi import MPPI 
from mppi_gps.envs.xarm7_env import Xarm7 
import time

env = Xarm7(render_mode="human")
planner_env = Xarm7()

# rollout states -> (K samples, T time steps, 53(state size))
def cost(states: Float[np.ndarray, "K T S"]) -> Float[np.ndarray, "K"]:
    # joint_pos = states[:, :, :7]
    joint = states[:, :, 2]
    dist = np.sqrt((-1.5-joint)**2)
    cost = np.sum(dist, axis=1)
    return cost

def pick_place_cost(states: Float[np.ndarray, "K T S"]) -> Float[np.ndarray, "K"]:
    distances = np.zeros([states.shape[0], states.shape[1]])
    for k in range(states.shape[0]):
        for t in range(states.shape[1]):
            state = states[k,t] 
            robot_qpos = state[1:9] 
            
            curr_qpos = planner_env.data.qpos.copy() 
            curr_qpos[:8] = robot_qpos

            planner_env.set_state(curr_qpos, planner_env.data.qvel)
            tcp_loc = planner_env.data.site("link_tcp").xpos
            target_loc = np.array([0.1, 0.4, 0.4]) 
            
            distances[k, t] = np.linalg.norm(target_loc - tcp_loc)
    costs = np.sum(distances, axis=1)
    return costs 


controller = MPPI(planner_env, pick_place_cost)
# farama env needs to reset before
env.reset()
for step in range(1000):
    state = np.concat([env.data.qpos, env.data.qvel])
    action = controller.action(state=state)

    env_action = action # action - env.data.qpos[:8]
    print(env.data.qpos[1])
    # controller.action != action in farama env 
    env.step(env_action)
    env.render()
    time.sleep(0.002)
