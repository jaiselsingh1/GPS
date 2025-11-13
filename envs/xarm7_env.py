import mujoco
import numpy as np
import gymnasium as gym 
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium import spaces
from dataclasses import dataclass



class Xarm7(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, model_path="models/assets/scene.xml", frame_skip=10, **kwargs):
        super().__init__(
            model_path, 
            frame_skip, 
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(65,), dtype=np.float32) # (qpos(13) + qvel(13) + (pose 7 (quat+xyz) + vel 6) = 13 each + ee_pos − can_pos (3), can_pos − place_site (3) → 6 + End-effector (TCP): pose 7 (quat+xyz) → 7 + Task relatives: ee_pos − can_pos (3), can_pos − place_site (3) → 6
            **kwargs, 
        )



