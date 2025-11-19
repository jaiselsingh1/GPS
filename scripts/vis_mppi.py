# import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"  # before importing jax or mjx

import mujoco 
import mujoco.mjx as mjx 
import jax 
import jax.numpy as jnp
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

def vec_pick_place_cost(states: Float[np.ndarray, "K T S"]) -> Float[np.ndarray, "K"]:
    K, T, S = states.shape 
    target_lift_height = 0.50 

    states_flat = states.reshape(K*T, S)
    nq = planner_env.model.nq 

    qpos_all = states_flat[:, 1:nq+1]

    # for the forward kinematics
    tcp_locs = np.zeros((K*T, 3))
    can_locs = np.zeros((K*T, 3))

    # you cannot do the numpy for the forward part with the simulator since you need to create multiple simulator instances 
    data_batch = compute_fk(qpos_all)

    tcp_site_id = mujoco.mj_name2id(planner_env.model, mujoco.mjtObj.mjOBJ_SITE, "link_tcp")
    can_body_id = mujoco.mj_name2id(planner_env.model, mujoco.mjtObj.mjOBJ_BODY, "can")
    
    tcp_locs = data_batch.site_xpos[:, tcp_site_id]  # (K*T, 3)
    can_locs = data_batch.xpos[:, can_body_id]  # (K*T, 3)

    # convert back to numpy arrays 
    tcp_locs = np.array(tcp_locs)
    can_locs = np.array(can_locs)

    dist_tcp_can = np.linalg.norm(can_locs - tcp_locs, axis=1)
    can_heights = can_locs[:, 2]
    lift_penalty = 100 * (target_lift_height - can_heights)**2

    costs_flat = dist_tcp_can + lift_penalty  # Shape: (K*T,)
    costs_per_dt = costs_flat.reshape(K, T)
    trajectory_costs = np.sum(costs_per_dt, axis=1)  # Shape: (K,)

    return trajectory_costs

mjx_model = mjx.put_model(planner_env.model)
@jax.jit 
def compute_fk(qpos_all):
    # qpos_all shape (K*T, S)
    # takes the function lambda and then applies it into every row of qpos_all 

    # the qpos_all in the second brackets is calling the vectorized function 
    data_batch = jax.vmap(lambda qpos: mjx.make_data(mjx_model))(qpos_all)
    
    # JAX requires immutability 
    data_batch = jax.vmap(lambda d,q: mjx.forward(mjx_model, d.replace(qpos=q)))(data_batch, qpos_all)
    return data_batch


def pick_place_cost(states: Float[np.ndarray, "K T S"]) -> Float[np.ndarray, "K"]:
    traj_costs = np.zeros([states.shape[0], states.shape[1]])
    target_lift_height = 0.50

    for k in range(states.shape[0]):
        for t in range(states.shape[1]):
            state = states[k,t] 
            # robot_qpos = state[1:9] 
            
            # curr_qpos = planner_env.data.qpos.copy() 
            curr_qpos = state[1 : planner_env.model.nq+1]

            planner_env.set_state(curr_qpos, planner_env.data.qvel)
            tcp_loc = planner_env.data.site("link_tcp").xpos

            can_loc = planner_env.data.body("can").xpos 
            box_loc = planner_env.data.body("box").xpos 

            dist_tcp_can = np.linalg.norm(can_loc - tcp_loc)

            can_height = can_loc[2]
            lift_penalty = 100 * (target_lift_height - can_height)**2
            print(can_height, target_lift_height)
            # target_loc = np.array([0.1, 0.4, 0.4])  # go to the top left 
            
            # joint_pen = 0.0001 * np.linalg.norm(state[1 + planner_env.model.nq :])

            print(dist_tcp_can, lift_penalty)
            traj_costs[k, t] = (dist_tcp_can + lift_penalty)

    costs = np.sum(traj_costs, axis=1)
    return costs 


controller = MPPI(planner_env, vec_pick_place_cost)
# farama env needs to reset before
env.reset()
for step in range(1000):
    state = np.concat([env.data.qpos, env.data.qvel])
    action = controller.action(state=state)

    env_action = action # action - env.data.qpos[:8]
    # controller.action != action in farama env 
    env.step(env_action)
    env.render()
    time.sleep(0.002)
