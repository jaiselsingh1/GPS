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
    def __init__(self, env, config: Optional[PickPlaceConfig] = None):
        self.env = env 
        self.config = config or PickPlaceConfig()

        # cache ids 
        self._sid_tcp   = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, env.tcp_site)
        self._sid_place = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, env.place_site)
        self._bid_can   = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, env.can_body)
        self._bid_box   = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, env.box_body)
        self._qadr_can, self._dadr_can = env._freejoint_addr(self._bid_can)
        self._qadr_box, self._dadr_box = env._freejoint_addr(self._bid_box)
        self._pick = self.cfg.pick_body # can or box 

    def reset(self):
        pass

        


            











