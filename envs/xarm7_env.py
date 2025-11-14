import mujoco
import numpy as np
import gymnasium as gym 
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium import spaces
from dataclasses import dataclass
import os

class Xarm7(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    can_body = "can"
    box_body = "box" 
    table_body = "table"
    pick_site = "pick_site"
    place_site = "place_site"
    tcp_site = "link_tcp"

    def __init__(self,
                 model_path: str = "models/assets/scene.xml",
                 frame_skip: int = 5,
                 ctrl_scale: float = 0.02,
                 include_tcp: bool = True,
                 include_rel: bool = True,
                 table_z: float = 0.38,
                 **kwargs):
        self.ctrl_scale  = float(ctrl_scale)
        self.include_tcp = bool(include_tcp)
        self.include_rel = bool(include_rel)
        self.table_z     = float(table_z)

        model_path = os.path.abspath(model_path)
        assert os.path.exists(model_path), f"model XML not found: {model_path}"

        super().__init__(model_path=model_path,
                         frame_skip=frame_skip,
                         default_camera_config=None,
                         observation_space=None,
                         **kwargs)
        
        # cache ids (sid = site id; bid = body id)
        self._sid_tcp   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.tcp_site)
        self._sid_place = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.place_site)
        self._bid_can   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.can_body)
        self._bid_box   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.box_body)
        self._qadr_can, self._dadr_can = self._freejoint_addr(self._bid_can)
        self._qadr_box, self._dadr_box = self._freejoint_addr(self._bid_box)
        self._arm_qpos_idx = self._find_arm_hinges(prefix="joint", count=7)

        # action = 7 dim delta q 
        self.act_dim = 7 
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)

    def step(self, action):
        a =  np.asarray(action, dtype=np.float64.clip(-1.0,1.0))
        dq = a * self.ctrl_scale 
        
        # desired dq 
        if not hasattr(self, "q_des"):
            self.q_des = self.data.qpos[self._arm_qpos_idx].copy()

        self.q_des = self.data.q_des + dq 

        # PD servo control ctrl = desired joint positions (MuJoCo applies torque = Kp(u-q) - Kd qdot)
        u = self.data.ctrl.copy()
        u[:self.act_dim] = self.q_des

        # advance physics 
        self.do_simulation(u, self.frame_skip)

        obs = self.get_current_obs()
        reward, info = self._default_reward()
        terminated = bool(info.get("success", False))
        truncated = False
        return obs, float(reward), terminated, truncated, info

    def reset_model(self, seed=None, options=None):
        # reset model to the initial pose (zero vel)
        self.init_qpos = getattr(self, "init_qpos", self.data.qpos.copy())
        self.init_qvel = getattr(self, "init_qvel", self.data.qvel.copy())
        self.set_state(self.init_qpos.copy(), np.zeros_like(self.init_qvel))

        # sync q_des to the current joint angles 
        self.q_des = self.data.qpos[self._arm_qpos_idx].copy()

        # randomize the locations of the objects 
        # rng = np.random.default_rng(seed)
        # self._spawn_free(self._qadr_can, self._dadr_can,
        #                  xyz=self._table_to_world([+0.12, 0.00, 0.41 - self.table_z]),
        #                  quat=[1,0,0,0], rng_xy=0.03, rng=rng)
        # self._spawn_free(self._qadr_box, self._dadr_box,
        #                  xyz=self._table_to_world([ 0.00, +0.18, 0.41 - self.table_z]),
        #                  quat=[1,0,0,0], rng_xy=0.03, rng=rng)

        # build the observation space dynamically 
        if self.observation_space is None:
            obs = self.get_current_obs()
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype=np.float32)

        # puts into the initial pose like gym and then returns the first observation
        return self.get_current_obs(), {}
    
    def get_current_obs(self):
        




        










    














"""
class Xarm7(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, model_path="models/assets/scene.xml", frame_skip=10, **kwargs):
        super().__init__(
            model_path, 
            frame_skip, 
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(65,), dtype=np.float32) # (qpos(13) + qvel(13) + (pose 7 (quat+xyz) + vel 6) = 13 each + ee_pos − can_pos (3), can_pos − place_site (3) → 6 + End-effector (TCP): pose 7 (quat+xyz) → 7 + Task relatives: ee_pos − can_pos (3), can_pos − place_site (3) → 6
            **kwargs, 
        )
"""


