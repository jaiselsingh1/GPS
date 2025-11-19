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
            cost: abc.Callable,  # feed the planner a cost function
            num_samples: int = 64, # number of samples 
            horizon: int = 16, # number of time steps 
            noise_sigma: float = 0.05, 
            lambda_: float = 0.00001,
    ):
        
        self.num_samples = num_samples
        self.horizon = horizon 
        self.noise_sigma = noise_sigma
        self.lambda_= lambda_
        self.cost = cost
        
        env.reset()
        self.model = env.model 
        self.data = env.data

        self.act_dim = env.action_space.shape[0]
        #env.action_space.shape[0]
        # control sequence over the horizon 
        self.U = np.zeros((self.horizon, self.act_dim))
        # start in the current state (initial controls = start of world)
        self.U[:,:] = self.data.qpos[:self.act_dim].copy()

    def action(self, state: Float[np.ndarray, "d"]) -> Float[np.ndarray, "a"]:
        lam = self.lambda_

        states = np.repeat(state[None], self.num_samples, axis=0)
        states = np.concat([np.zeros((self.num_samples, 1)), states], axis=-1)

        noise = np.random.randn(self.num_samples, self.horizon, self.act_dim) * self.noise_sigma # (K, T, U)
        controls = self.U[None] + noise # same as self.U[np.newaxis] or self.U.reshape(1, self.horizon, self.act_dim)
        
        rollout_states, _ = rollout.rollout(self.model, self.data, states, controls, persistent_pool=True)
        costs = self.cost(rollout_states)
        # minimum cost 
        beta = np.min(costs)
    
        exp_costs = np.exp(-1.0 * (costs - beta)/self.lambda_)
        # eta is the normalizer which is just a weighted cost averaging (think sorta softmax)
        eta = np.sum(exp_costs)
        weights = exp_costs / eta
        # print(np.max(weights))

        weighted_noise = np.sum(noise * weights[:, None, None], axis=0)
        updated_controls = self.U + weighted_noise

        self.U[:self.horizon-1,:] = updated_controls[1:,:]
        self.U[self.horizon-1, :] = updated_controls[self.horizon-1, :]

        return updated_controls[0]