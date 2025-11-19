import os
import mujoco
import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium import spaces
from dataclasses import dataclass
import pathlib

# never use relative directories where you aren't sure which folder you will be running from (anchor from relative path that you are certain of)
cur_path = pathlib.Path(__file__) # __file__ is always set to the absolute dir 
mppi_gps_dir = cur_path.parent.parent 

from mppi_gps.tasks.pick_place import PickPlaceTask, PickPlaceConfig

class Xarm7(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    can_body = "can"
    box_body = "box" 
    table_body = "table"
    pick_site = "pick_site"
    place_site = "place_site"
    tcp_site = "link_tcp"

    def __init__(self,
                 model_path: str = "models/assets/scene.xml",
                 frame_skip: int = 5,
                 ctrl_scale: float = 0.01,
                 gr_ctrl_scale: float = 255.0,
                 include_tcp: bool = True,
                 include_rel: bool = True,
                 table_z: float = 0.38,
                 **kwargs):
        self.ctrl_scale  = float(ctrl_scale)
        self.gr_ctrl_scale = float(gr_ctrl_scale)
        self.include_tcp = bool(include_tcp)
        self.include_rel = bool(include_rel)
        self.table_z     = float(table_z)
        self.observation_space = None

        model_path = mppi_gps_dir/model_path
        assert os.path.exists(model_path), f"model XML not found: {model_path}"

        super().__init__(model_path=model_path.as_posix(), #as_posix() to make into str
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

        # action = 7 dim delta q + gripper
        self.act_dim = 8
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)

    def step(self, action):
        # a  = np.asarray(action, dtype=np.float64).clip(-1.0, 1.0)
        # dq = a * self.ctrl_scale 
        # dq[7] = a[7] * self.gr_ctrl_scale
        
        # self.q_des = self.q_des + dq 

        # PD servo control ctrl = desired joint positions (MuJoCo applies torque = Kp(u-q) - Kd qdot)
        # u = self.data.ctrl.copy()
        # u[:self.act_dim] = self.q_des

        # advance physics 
        self.do_simulation(action, 1) #self.frame_skip)

        obs = self.get_current_obs()
        reward, info = self._default_reward()
        terminated = bool(info.get("success", False))
        truncated = False
        return obs, float(reward), terminated, truncated, info

    def reset_model(self, seed=None, options=None):
        # reset model to the initial pose (zero vel)
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()
        # based on vis in the simulator
        self.init_qpos[1] = -0.5 
        self.set_state(self.init_qpos.copy(), np.zeros_like(self.init_qvel))

        if not hasattr(self, "task"):
            self.task = PickPlaceTask(self, PickPlaceConfig())
        # self.task.reset(seed=seed, randomize=False)

        # sync q_des to the current joint angles 
        self.q_des = self.data.qpos[:self.act_dim].copy()

        # build the observation space dynamically 
        if self.observation_space is None:
            obs = self.get_current_obs()
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype=np.float32)

        # puts into the initial pose like gym and then returns the first observation
        return self.get_current_obs() 
    
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
            tcp_p = self.data.site_xpos[self._sid_tcp].copy()
            tcp_quat = np.empty(4, dtype=np.float64)
            mat9 = np.asarray(self.data.site_xmat[self._sid_tcp], dtype=np.float64)  # length-9
            mujoco.mju_mat2Quat(tcp_quat, mat9) # [w, x, y, z]
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
        # joints attached to this body are in [j0, j0 + jn)
        j0 = int(self.model.body_jntadr[body_id])
        jn = int(self.model.body_jntnum[body_id])
        if jn == 0:
            raise RuntimeError("Body has no joints")

        for j in range(j0, j0 + jn):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                qadr = int(self.model.jnt_qposadr[j])  # start of 7-d pose (quat+xyz)
                dadr = int(self.model.jnt_dofadr[j])   # start of 6-d vel (ang+lin)
                return qadr, dadr
        raise RuntimeError("Body has no freejoint")

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
