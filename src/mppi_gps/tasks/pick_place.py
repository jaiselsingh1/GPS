from __future__ import annotations 
import mujoco
import numpy as np
from dataclasses import dataclass 
from typing import Optional

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
        self.cfg = config or PickPlaceConfig()  # <-- keep as cfg consistently

        # cache ids 
        self._sid_tcp   = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, env.tcp_site)
        self._sid_place = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, env.place_site)
        self._bid_can   = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, env.can_body)
        self._bid_box   = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, env.box_body)
        self._qadr_can, self._dadr_can = env._freejoint_addr(self._bid_can)
        self._qadr_box, self._dadr_box = env._freejoint_addr(self._bid_box)
        self._pick = self.cfg.pick_body # can or box 

    def reset(self, seed: Optional[int] = None, randomize: bool = True):
        rng = np.random.default_rng(seed)
        table_z = self.env.table_z

        # default table-relative spawns
        can_rel = np.array([+0.12,  0.00, 0.41 - table_z])
        box_rel = np.array([ 0.00, +0.18, 0.41 - table_z])

        if randomize:
            can_rel[:2] += rng.uniform(-self.cfg.can_xy_noise, self.cfg.can_xy_noise, size=2)
            box_rel[:2] += rng.uniform(-self.cfg.box_xy_noise, self.cfg.box_xy_noise, size=2)

        self._spawn(self._qadr_can, self._dadr_can,
                    xyz=self.env._table_to_world(can_rel), quat=[1,0,0,0])
        self._spawn(self._qadr_box, self._dadr_box,
                    xyz=self.env._table_to_world(box_rel), quat=[1,0,0,0])

    # metrics / reward used by env 
    def _metrics(self):
        d = self.env.data
        tcp   = d.site_xpos[self._sid_tcp].copy()
        goal  = d.site_xpos[self._sid_place].copy()
        can_p = d.qpos[self._qadr_can+4 : self._qadr_can+7].copy()
        box_p = d.qpos[self._qadr_box+4 : self._qadr_box+7].copy()
        obj   = can_p if self._pick == "can" else box_p
        d_reach = float(np.linalg.norm(tcp - obj))
        d_place = float(np.linalg.norm(obj - goal))
        lifted  = bool(obj[2] > (self.env.table_z + self.cfg.lift_height))
        return dict(tcp=tcp, goal=goal, can_pos=can_p, box_pos=box_p,
                    obj_pos=obj, d_reach=d_reach, d_place=d_place, lifted=lifted)

    def reward(self):
        m = self._metrics()
        r = 0.5*np.exp(-self.cfg.reach_alpha * m["d_reach"]) + 0.5*np.exp(-self.cfg.place_alpha * m["d_place"])
        if m["lifted"]:
            r += self.cfg.lift_bonus
        success = (m["d_place"] < self.cfg.tol_xy) and m["lifted"]
        info = dict(success=bool(success), **m, pick=self._pick)
        return float(r), info

    def cost(self):
        r, _ = self.reward()
        return float(-r)

    # spawn helper / internal

    def _spawn(self, qadr: int, dadr: int, *, xyz, quat):
        q = np.asarray(quat, dtype=np.float64)
        q = q / max(np.linalg.norm(q), 1e-9)
        p = np.asarray(xyz, dtype=np.float64)
        self.env.data.qpos[qadr:qadr+4]   = q
        self.env.data.qpos[qadr+4:qadr+7] = p
        self.env.data.qvel[dadr:dadr+6]   = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
