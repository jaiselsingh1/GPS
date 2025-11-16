import os
import mujoco
import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium import spaces
from dataclasses import dataclass

from tasks.pick_place import PickPlaceTask, PickPlaceConfig

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
        a  = np.asarray(action, dtype=np.float64).clip(-1.0, 1.0)
        dq = a * self.ctrl_scale 
        
        # desired dq 
        if not hasattr(self, "q_des"):
            self.q_des = self.data.qpos[self._arm_qpos_idx].copy()

        self.q_des = self.q_des + dq 

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

        if not hasattr(self, "task"):
            self.task = PickPlaceTask(self, PickPlaceConfig())
        self.task.reset(seed=seed, randomize=True)

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
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        # can/box free joint states --> pose(7)=quat+xyz, vel(6)=ang+lin
        # q is the pose whereas d is the velocity components 
        can_q = self.data.qpos[self._qadr_can:self._qadr_can+7]
        can_d = self.data.qvel[self._dadr_can:self._dadr_can+6]
        box_q = self.data.qpos[self._qadr_box:self._qadr_box+7]
        box_d = self.data.qvel[self._dadr_box:self._dadr_box+6]

        obs_comp = [qpos, qvel, can_q, can_d, box_q, box_d]

        if self.include_tcp:
            tcp_p = self.data.site_xpos[self._sid_tcp].copy() # (3,)
            tcp_quat = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(tcp_quat, self.data.site_xmat[self._sid_tcp].reshape(3,3)) # mujoco uses [w, x, y, z]
            obs_comp += [tcp_quat, tcp_p]

        if self.include_rel:
            can_p = can_q[3:6] # can x, y and z 
            place_p = self.data.site_xpos[self._sid_place].copy()     # place site position
            tcp_p   = self.data.site_xpos[self._sid_tcp].copy()       # tcp position
            obs_comp += [tcp_p - can_p, can_p - place_p] # vector from can to TCP

        return np.concatenate([p.ravel() for p in obs_comp]).astype(np.float32)
                        
        # flatten returns a copy 
        # ravel returns a view 

    def _default_reward(self):
        return self.task.reward()
        # tcp_p   = self.data.site_xpos[self._sid_tcp].copy()
        # place_p = self.data.site_xpos[self._sid_place].copy()
        # can_p   = self.data.qpos[self._qadr_can+4 : self._qadr_can+7].copy()

        # d_reach = float(np.linalg.norm(tcp_p - can_p))
        # d_place = float(np.linalg.norm(can_p - place_p))

        # r = 0.5*np.exp(-4.0*d_reach) + 0.5*np.exp(-4.0*d_place)
        # lifted = can_p[2] > (self.table_z + 0.02)
        # if lifted:
        #     r += 0.25

        # success = (d_place < 0.03) and lifted
        # info = dict(success=bool(success), d_reach=d_reach, d_place=d_place, lifted=lifted)
        # return float(r), info

    # helpers 
    def _find_arm_hinges(self, prefix: str, count: int):
        idxs = []
        for j in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if name.startswith(prefix) and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                idxs.append(int(self.model.jnt_qposadr[j]))
        if len(idxs) < count:
            raise RuntimeError(f"Found {len(idxs)} hinge joints with prefix '{prefix}', need {count}")
        return np.array(idxs[:count], dtype=np.int64)

    def _freejoint_addr(self, body_id: int):
        jid = self.model.body_jntid[body_id]
        if jid < 0 or self.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("Body has no freejoint")
        return int(self.model.jnt_qposadr[jid]), int(self.model.jnt_dofadr[jid])

    def _spawn_free(self, qadr, dadr, *, xyz, quat, rng_xy, rng):
        noise = np.array([rng.uniform(-rng_xy, rng_xy),
                          rng.uniform(-rng_xy, rng_xy), 0.0])
        p = np.asarray(xyz, dtype=np.float64) + noise
        q = np.asarray(quat, dtype=np.float64); q /= max(np.linalg.norm(q), 1e-9)
        self.data.qpos[qadr:qadr+4] = q
        self.data.qpos[qadr+4:qadr+7] = p
        self.data.qvel[dadr:dadr+6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _table_to_world(self, rel_xyz):
        tbid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.table_body)
        return self.model.body_pos[tbid].copy() + np.asarray(rel_xyz, dtype=np.float64)
