import mujoco_warp as mjw 
import mujoco 
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from dataclasses import dataclass
import warp as wp
from jaxtyping import Float, Array

@dataclass 
class MjwData:
    model: mjw.Model
    data: mjw.Data
    K: int 
    T: int
    

def make_data(env: MujocoEnv, K: int, T: int) -> MjwData:
    data = MjwData(
        model = mjw.put_model(env.model), 
        data = mjw.make_data(env.model, K),
        K = K, 
        T = T
    )

    return data


def mjw_rollout(mjw_data: MjwData, init_state: Float[Array, "d"], controls: Float[Array, "K T a"]) -> Float[Array, "K T+1 d"]:
    K, T = mjw_data.K, mjw_data.T
    init_qpos = init_state[: mjw_data.model.nq]
    init_qvel = init_state[mjw_data.model.nq :]

    wp.copy(mjw_data.data.qpos, wp.array([init_qpos for k in range(K)],dtype=float)) # doing k parallel rollouts 
    wp.copy(mjw_data.data.qvel, wp.array([init_qvel for k in range(T)],dtype=float))

    d = init_state.shape[0]
    rollout_states = wp.zeros((K, T+1, d), dtype=float)

    wp_controls = wp.array(controls, dtype=wp.float32)
    
    for t in range(T):
        mjw_data.data.ctrl = wp_controls[t]
        
        mjw.step(mjw_data.model, mjw_data.data)
        qpos = mjw_data.data.qpos 
        qvel = mjw_data.data.qvel
        wp.copy(rollout_states[t,: mjw_data.model.nq], qpos)
        wp.copy(rollout_states[t, mjw_data.model.nq: ], qvel)

    return rollout_states


def capture_rollout(mjw_data: MjwData, init_state: Float[Array, "d"], controls: Float[Array, "K T a"]) -> wp.ScopedCapture:
    with wp.ScopeCapture() as capture:
        K, T = mjw_data.K, mjw_data.T
        init_qpos = init_state[: mjw_data.model.nq]
        init_qvel = init_state[mjw_data.model.nq :]

        wp.copy(mjw_data.data.qpos, wp.array([init_qpos for k in range(K)],dtype=float)) # doing k parallel rollouts 
        wp.copy(mjw_data.data.qvel, wp.array([init_qvel for k in range(T)],dtype=float))

        d = init_state.shape[0]
        rollout_states = wp.zeros((K, T+1, d), dtype=float)

        wp_controls = wp.array(controls)
        
        for t in T:
            mjw_data.data.ctrl = wp_controls[t]

            mjw.step(mjw_data.model, mjw_data.data)
            qpos = mjw_data.data.qpos 
            qvel = mjw_data.data.qvel
            rollout_states[t,: mjw_data.model.nq] = qpos 
            rollout_states[t, mjw_data.model.nq: ] = qvel


