import torch
import mujoco
from mujoco import rollout
from gymnasium import spaces 
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from jaxtyping import Float
import numpy as np
import typing
from collections import abc

class MPPI:
    def __init__(
            self, 
            env: MujocoEnv, 
            cost: abc.Callable, 
            num_samples: int = 5, # number of samples 
            horizon: int = 10, # number of time steps 
            noise_sigma: float = 0.2, 
            phi: float = None, 
            q: float = None, 
            lambda_: float = 100.0,
    ):
        
        self.num_samples = num_samples
        self.horizon = horizon 
        self.noise_sigma = noise_sigma
        self.phi = phi 
        self.q = q
        self.lambda_= lambda_
        
        env.reset()
        self.model = env.model 
        self.data = env.data

        self.act_dim = env.action_space.shape[0]
        #env.action_space.shape[0]
        # control sequence over the horizon 
        self.U = np.zeros((self.horizon, self.act_dim))

    def action(self, state: Float[np.ndarray, "d"]) -> Float[np.ndarray, "a"]:
        lam = self.lambda_
        
        states = np.repeat(state[None], self.num_samples, axis=0)
        states = np.concat([np.zeros((self.num_samples, 1)), states], axis=-1)

        noise = np.random.randn(self.num_samples, self.horizon, self.act_dim) * self.noise_sigma
        controls = self.U[None] + noise
        
        rollout_states, _ = rollout.rollout(self.model, self.data, states, controls, persistent_pool=True)
        

