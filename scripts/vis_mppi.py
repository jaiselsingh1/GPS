import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"  
os.environ["MUJOCO_GL"] = "egl"
import mujoco 
import mujoco.mjx as mjx 
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
import jax 
import jax.numpy as jnp
from jaxtyping import Float, Array
import numpy as np
from mppi_gps.controllers.jax_mppi import MPPI_JAX
from mppi_gps.controllers.mppi import MPPI
from mppi_gps.envs.xarm7_env import Xarm7
from typing import Callable
from imageio import v3 
import tqdm
import time

env = Xarm7(render_mode="rgb_array")
planner_env = Xarm7()

_mjx_model = mjx.put_model(planner_env.model) # device = jax.devices("cpu")[0])
nq = planner_env.model.nq
_mjx_data = mjx.make_data(_mjx_model)
_tcp_sid   = mujoco.mj_name2id(planner_env.model, mujoco.mjtObj.mjOBJ_SITE, "link_tcp")
_can_bid   = mujoco.mj_name2id(planner_env.model, mujoco.mjtObj.mjOBJ_BODY, "can")
_can_jid   = planner_env.model.body_jntadr[_can_bid]
_can_qadr  = planner_env.model.jnt_qposadr[_can_jid]  # start of can's free-joint in qpos

def _fk_one(q: Float[Array, "nq"]) -> Float[Array, "3"]:
    d = _mjx_data.replace(qpos=q)
    d = mjx.kinematics(_mjx_model, d)
    # d = mjx.forward(_mjx_model,d)
    return d.site_xpos[_tcp_sid]

_fk_tcp_batch = jax.vmap(_fk_one)

@jax.jit
def vec_pick_place_cost(states: Float[Array, "K T S"]) -> Float[Array, "K"]:
    K, T, S = states.shape
    states_flat = states.reshape((K*T, S))
    qpos_all = states_flat[:, 1:nq+1]

    tcp_locs = _fk_tcp_batch(qpos_all)
    can_locs = qpos_all[:, _can_qadr: _can_qadr + 3]

    dist_tcp_can = jnp.linalg.norm(can_locs - tcp_locs, axis=1)

    can_heights = can_locs[:, 2]
    lift_penalty = 100.0 * (target_lift_height - can_heights) ** 2

    costs_flat = dist_tcp_can # + lift_penalty 
    costs = costs_flat.reshape((K, T)).sum(axis=1)
    return costs

# building a cost function that can be passed into the MPPI planner
def make_cost_jax(
    planner_env: MujocoEnv, 
    target_lift_height: float = 0.50) -> Callable[[Float[Array, "K T S"]], Float[Array, "K"]]:
    
    mjx_model = mjx.put_model(planner_env.model)

    # pre compute the indices 
    tcp_sid: int = mujoco.mj_name2id(planner_env.model, mujoco.mjtObj.mjOBJ_SITE, "link_tcp")
    can_bid: int = mujoco.mj_name2id(planner_env.model, mujoco.mjtObj.mjOBJ_BODY, "can")
    can_jid: int = planner_env.model.body_jntadr[can_bid]
    can_qadr: int = planner_env.model.jnt_qposadr[can_jid]  # start of can's free joint

    data_template = mjx.make_data(mjx_model)
    nq = mjx_model.nq

    @jax.jit
    def fk_tcp_batch(qpos_flat: Float[Array, "N nq"]) -> Float[Array, "N 3"]:
        # batched FK for the TCP site 

        @jax.vmap
        def fk_one(q: Float[Array, "nq"]) -> Float[Array, "3"]:
            d = data_template.replace(qpos=q)
            d = mjx.forward(mjx_model, d)
            return d.site_xpos[tcp_sid]
        
        return fk_one(qpos_flat)
    
    @jax.jit
    def cost(states: Float[Array, "K T S"]) -> Float[Array, "K"]:
        K, T, S = states.shape

        qpos: Float[Array, "K T nq"] = states[:, :, :nq]
        qpos_flat: Float[Array, "KT nq"] = qpos.reshape(-1, nq)

        tcp_locs_flat = fk_tcp_batch(qpos_flat)
        can_locs_flat = qpos_flat[:, can_qadr : can_qadr + 3]

        dist_tcp_can = jnp.linalg.norm(can_locs_flat - tcp_locs_flat, axis=1)
        can_heights = can_locs_flat[:, 2]
        height_diff = target_lift_height - can_heights
        lift_penalty = 100.0 * height_diff * height_diff

        costs_flat = dist_tcp_can + lift_penalty
        costs_per_traj = costs_flat.reshape(K, T)
        return jnp.sum(costs_per_traj, axis=1)
    
    return cost

target_lift_height = 0.50
cost_jax = make_cost_jax(planner_env, target_lift_height=target_lift_height)
# controller = MPPI(planner_env, vec_pick_place_cost)
controller = MPPI_JAX(planner_env, cost_jax)

# farama env needs to reset before
env.reset()
frames = []

for step in tqdm.tqdm(range(1000)):
    state = np.concatenate([env.data.qpos, env.data.qvel])
    action = controller.action(state=state)

    env_action = action # action - env.data.qpos[:8]
    # controller.action != action in farama env 
    env.step(env_action)
    # env.render()
    frames.append(env.render())
    print("render complete")
    # time.sleep(0.002)
v3.imwrite("jax_pick_place.mp4", frames, fps=250)