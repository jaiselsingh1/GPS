import os
os.environ["MUJOCO_GL"] = "egl"

import time
import numpy as np
import jax
import mujoco
import mujoco.mjx as mjx
from mppi_gps.controllers.jax_mppi import MPPI_JAX
from mppi_gps.controllers.mppi import MPPI

from mppi_gps.envs.xarm7_env import Xarm7

print(f"JAX devices: {jax.devices()}")

env = Xarm7(render_mode="rgb_array")
planner_env = Xarm7()

# Simple cost function - just distance, no FK
nq = planner_env.model.nq

@jax.jit
def simple_cost(states):
    K, T, S = states.shape
    # Just sum of squared positions - dummy cost
    return jax.numpy.sum(states[:, :, :nq] ** 2, axis=(1, 2))

print("Creating controller...")
t0 = time.time()
controller = MPPI(planner_env, simple_cost, num_samples=8, horizon=4)
print(f"Init + compile: {time.time() - t0:.2f}s")

env.reset()

# Warmup
state = np.concatenate([env.data.qpos, env.data.qvel])
t0 = time.time()
_ = controller.action(state=state)
print(f"First action: {time.time() - t0:.4f}s")

# Time loop without rendering
print("\nTiming 50 steps (no render):")
t0 = time.time()
for step in range(50):
    state = np.concatenate([env.data.qpos, env.data.qvel])
    action = controller.action(state=state)
    env.step(action)
print(f"50 steps: {time.time() - t0:.2f}s ({(time.time() - t0) / 50 * 1000:.1f}ms per step)")

# Time with rendering
print("\nTiming 50 steps (with render):")
t0 = time.time()
for step in range(50):
    state = np.concatenate([env.data.qpos, env.data.qvel])
    action = controller.action(state=state)
    env.step(action)
    _ = env.render()
print(f"50 steps: {time.time() - t0:.2f}s ({(time.time() - t0) / 50 * 1000:.1f}ms per step)")