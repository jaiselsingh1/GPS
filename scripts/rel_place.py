import numpy as np 
from mppi_gps.envs.xarm7_env import Xarm7 
import mujoco 

def get_base_ids(env):
    bid_base = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "link_base")
    assert bid_base >= 0, "Could not find body 'link_base' in model"
    return bid_base 

def base_pose_world(env):
    bid = get_base_ids(env)
    R = env.data.xmat[bid].reshape(3,3).copy() # R is in SO(3) in world coordinates
    x = env.data.xpos[bid].copy() # x in R^3 both are in world coordinates 
    return R, x

def base_to_world(env, p_base):
    # construct a transformation to go from the base to the world frame for a 3d point
    R, x = base_pose_world(env)
    # R here is base in world coordinates, and x is the position of the base from the world origin 
    p_world =  x + R @ p_base 
    return p_world

# setting the pose of the object that you specify where the qvel = 0 so it is not moving 
def set_free_pose(env, body_name, *, xyz_world, quat_world=(1,0,0,0)):
    bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    assert bid >= 0, f"Body '{body_name}' not found"

    









