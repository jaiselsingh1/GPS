from __future__ import annotations 
import mujoco
from dataclasses import dataclass 
from typing import Dict, Tuple, Optional
import numpy as np
import os 

# tcp stands for tool center point 

@dataclass 
class PickPlaceConfig:
    name: str = "pick place object"
    # tolerances for the pick and place 
    tol_xy: float = 0.03 
    tol_z: float = 0.03 

    # reward strength scaling 
    reach_alpha: float = 4.0 # EE to the can 
    place_alpha: float = 4.0 # can to goal 

    lift_height: float = 0.02 # minimum distance to call something "lifted"
    lift_bonus: float = 0.25 

    # for domain randomization 
    can_xy_noise: float = 0.0 
    box_xy_noise: float = 0.0 

    # which object to pick/place
    pick_body: str = "can"


class PickPlaceTask:
    def __init__(self, env, config=Optional[PickPlaceConfig] = None):
        self.env = env 
        self.config = config or PickPlaceConfig()

        self.can = env.can_body
        self.box = env.box_body 
        self.tcp_site = env.tcp_site  # tool center point 
        
        def __init__(self, 
                     model_path="models/assets/scene.xml", 
                     frame_skip=10, 
                     ctrl_scale=0.02, 
                     include_tcp=True, 
                     include_rel=True, 
                     table_z=0.38, 
                     **kwargs):
            
            self.ctrl_scale = float(ctrl_scale)
            self.include_tcp = bool(include_tcp)
            self.include_rel = bool(include_rel)
            self.table_z     = float(table_z)

            model_path = os.path.abspath(model_path)
            assert os.path.exists(model_path), f"model XML not found: {model_path}"

            super().__init__(
                model_path = model_path, 
                frame_skip = frame_skip,
                default_camera_config = None, 
                observation_space = None
            )

            # cache ids 
            self._sid_tcp = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.tcp_site)
            self._sid_place = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.place_site)
            self._bid_can   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.can_body)
            self._bid_box   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.box_body)
            self._qadr_can, self._dadr_can = self._freejoint_addr(self._bid_can)
            self._qadr_box, self._dadr_box = self._freejoint_addr(self._bid_box)
            self._arm_qpos_idx = self._find_arm_hinges(prefix="joint", count=7)

            # action: 7 dim delta q 
            self.act_dim = 7 
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)
            


            











