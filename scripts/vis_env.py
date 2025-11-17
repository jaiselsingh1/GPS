import mujoco
import numpy as np
import time
from mppi_gps.envs.xarm7_env import Xarm7

env = Xarm7(render_mode="human")
steps = 1000
obs, _ = env.reset()

action = np.zeros_like(env.action_space.sample())
for i in range(steps):
    if i % 10 == 0:
        action[7] = 255.0
    env.step(action)
    time.sleep(0.002)
    env.render()
