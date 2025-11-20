import jax 
import jax.numpy as jnp 
import mujoco 
import mujoco.mjx as mjx 
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from jaxtyping import Float 
import numpy as np
from collections import abc 

class MPPI_JAX:
    def __init__(
        self, 
        env: MujocoEnv,
        cost: abc.Callable,  # Cost function that takes JAX array as input (aka states as jnp)
        num_samples: int = 64,
        horizon: int = 16,
        noise_sigma: float = 0.01,
        lambda_: float = 0.00001,
        frame_skip: int = 5,
    ):
        self.num_samples = num_samples 
        self.horizon = horizon 
        self.noise_sigma = noise_sigma 
        self.lambda_ = lambda_
        self.frame_skip = frame_skip

        env.reset()
        self.mjx_model = mjx.put_model(env.model)
        self.act_dim = env.action_space.shape[0]

        init_control = jnp.array(self.act_dim)
        # tile repeats along the current dimensions 
        self.U = jnp.tile(init_control, (self.horizon, 1)) # self.horizon, 1 tells you how to repeat along each direction 

        # random key for JAX (needed for when you want to do JIT)
        self.key = jax.random.PRNGKey(0)

        self._compile_rollout()

    def _compile_rollout(self) -> None:
        dummy_state = jnp.zeros(self.mjx_model.nq + self.mjx_model.nv)
        dummy_controls = jnp.zeros((self.num_samples, self.horizon * self.frame_skip, self.act_dim))
        _ = self._rollout_trajectories(dummy_state, dummy_controls)

    @jax.jit 
    def _rollout_trajectories(self, 
                              state: Float[jnp.array, "d"], 
                              controls: Float[jnp.array, "K T a"]) -> Float[jnp.array, "K T+1 d"]:
        K = controls.shape[0]
        T = controls.shape[1]
        nq = self.mjx_model.nq 
        nv = self.mjx_model.nv 

        # initialize data 
        qpos_init = state[:nq]
        qvel_init = state[nq:]

        def init_data(i):
            data = mjx.make_data(self.mjx_model)
            data = data.replace(qpos=qpos_init, qvel=qvel_init)
            return data 
        
        data_batch = jax.vmap(init_data)(jnp.arange(K))

        def step_fn(data, ctrl):
            data = data.replace(ctrl=ctrl)
            data = mjx.step(self.mjx_model, ctrl)
            # including time for now 
            state = jnp.concatenate([[data.time], data.qpos, data.qvel])
            return data, state
        
        





        

        

    


        
