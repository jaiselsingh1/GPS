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

controller = MPPI(planner_env, cost)
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
