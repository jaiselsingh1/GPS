import jax 
import jax.numpy as jnp 
import mujoco 
import mujoco.mjx as mjx 
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from jaxtyping import Float, PRNGKeyArray, Array
from typing import Tuple
import numpy as np
from collections import abc 
from functools import partial # this is needed to avoid passing in self as an array into a jitted function

class MPPI_JAX:
    def __init__(
        self, 
        env: MujocoEnv,
        cost: abc.Callable,  # Cost function that takes JAX array as input (aka states as jnp)
        num_samples: int = 64,
        horizon: int = 16,
        noise_sigma: float = 0.01,
        lambda_: float = 0.1,
        frame_skip: int = 5,
    ):
        self.num_samples = num_samples 
        self.horizon = horizon 
        self.noise_sigma = noise_sigma 
        self.lambda_ = lambda_
        self.frame_skip = frame_skip
        self.cost = cost

        env.reset()
        self.mjx_model = mjx.put_model(env.model)
        self.act_dim = env.action_space.shape[0]

        init_control = jnp.zeros(self.act_dim)
        # tile repeats along the current dimensions 
        self.U = jnp.tile(init_control, (self.horizon, 1)) # self.horizon, 1 tells you how to repeat along each direction 

        # random key for JAX (needed for when you want to do JIT)
        self.key = jax.random.PRNGKey(0)

        self._compile_rollout()

    def _compile_rollout(self) -> None:
        dummy_state = jnp.zeros(self.mjx_model.nq + self.mjx_model.nv)
        dummy_controls = jnp.zeros((self.num_samples, self.horizon * self.frame_skip, self.act_dim))
        _ = self._rollout_trajectories(dummy_state, dummy_controls)

        dummy_U = jnp.zeros((self.horizon, self.act_dim))
        dummy_key = jax.random.PRNGKey(0)
        _ = self._mppi_step(dummy_state, dummy_U, dummy_key)

    @partial(jax.jit, static_argnums=0)
    def _rollout_trajectories(self, 
                              state: Float[Array, "d"], 
                              controls: Float[Array, "K T a"]) -> Float[Array, "K T+1 d"]:
        K, T, _ = controls.shape
        nq = self.mjx_model.nq 
        nv = self.mjx_model.nv 

        # initialize data 
        qpos_init = state[:nq]
        qvel_init = state[nq:]

        data0 = mjx.make_data(self.mjx_model)
        data0 = data0.replace(qpos=qpos_init, qvel=qvel_init)

        def step_fn(data, ctrl):
            data = data.replace(ctrl=ctrl)
            data = mjx.step(self.mjx_model, data)
            state_vec = jnp.concatenate([data.qpos, data.qvel])
            return data, state_vec
        
        # vectorize over K trajectories 
        def rollout_one_traj(traj_ctrl):
            # traj_ctrl is control for one traj (T x act_dim)
            _, states = jax.lax.scan(step_fn, data0, traj_ctrl) # (function, initial value, sequence of inputs)

            init_state = jnp.concatenate([data0.qpos, data0.qvel])
            # prepend the initial state 
            states = jnp.concatenate([init_state[None], states], axis=0)
            return states
        
        # rollout all K samples 
        states = jax.vmap(rollout_one_traj)(controls)

        return states 
    
    @partial(jax.jit, static_argnums=0)
    def _mppi_step(
        self, 
        state: Float[Array, "d"], 
        U: Float[Array, "T a"], 
        key: PRNGKeyArray) -> Tuple[Float[Array, "T a"], Float[Array, "a"]]:
        # returns the updated control sequence and the next action to take

        # generate noise 
        noise = jax.random.normal(key, (self.num_samples, self.horizon, self.act_dim)) * self.noise_sigma

        # perturbed controls 
        controls = U[None, :, :] + noise # K, T, A dim 

        # repeat the controls for frame skip 
        controls_repeated = jnp.repeat(controls, self.frame_skip, axis=1)

        # rollout the trajectories 
        rollout_states_repeated = self._rollout_trajectories(state, controls_repeated)
        rollout_states = rollout_states_repeated[:, ::self.frame_skip, :]

        # compute cost 
        costs = self.cost(rollout_states)

        # weighted costs 
        beta = jnp.min(costs)
        exp_costs = jnp.exp(-(costs - beta) / self.lambda_)
        weights = exp_costs / jnp.sum(exp_costs)

        # weighted update
        weighted_noise = jnp.einsum('k,kta->ta', weights, noise)
        updated_controls = U + weighted_noise

        # shift the control sequence 
        new_U = jnp.concatenate([
            updated_controls[1:, :], # drop the first control
            updated_controls[-1:, :]], # repeat the last control (the [-1:, :] is to get the last row but keep it as it's original dim)
            axis=0)
        
        action = updated_controls[0]
        return new_U, action
    
    def action(self, state: Float[np.ndarray, "d"]) -> Float[np.ndarray, "a"]:
        state_jax = jnp.array(state)
        self.key, subkey = jax.random.split(self.key)

        # run the MPPI step 
        new_U, action = self._mppi_step(state_jax, self.U, subkey)
        # update control seq 
        self.U = new_U 
        action_np = np.array(action)

        return action_np