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

    
    


        
