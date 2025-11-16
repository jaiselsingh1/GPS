import mujoco
import numpy as np
from mppi_gps.envs.xarm7_env import Xarm7

env = Xarm7(render_mode="human")
steps = 100
obs, _ = env.reset()

action = np.zeros_like(env.action_space.sample())
for i in range(steps):
    env.step(action)
    env.render()
