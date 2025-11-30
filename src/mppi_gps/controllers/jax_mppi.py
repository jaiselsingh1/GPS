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
        num_samples: int = 16,
        horizon: int = 8,
        noise_sigma: float = 0.01,
        lambda_: float = 0.00001,
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
        
        # Pre-allocate data structure once; this way you don't initialize the mjdata at each step
        self.mjx_data_template = mjx.make_data(self.mjx_model) 
        
        # Pre-compute indices for state extraction
        self.nq = self.mjx_model.nq
        self.nv = self.mjx_model.nv
        self.state_dim = self.nq + self.nv

        init_control = jnp.zeros(self.act_dim)
        # tile repeats along the current dimensions 
        self.U = jnp.tile(init_control, (self.horizon, 1)) # self.horizon, 1 tells you how to repeat along each direction 

        # random key for JAX (needed for when you want to do JIT)
        self.key = jax.random.PRNGKey(0)

        self._compile_rollout()

    def _compile_rollout(self) -> None:
        dummy_state = jnp.zeros(self.state_dim)
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

        # initialize data 
        qpos_init = state[:self.nq]
        qvel_init = state[self.nq:]

        # vectorize over K trajectories
        @jax.vmap
        def rollout_one_traj(traj_ctrl):
            # traj_ctrl is control for one traj (T x act_dim)
            data = self.mjx_data_template.replace(qpos=qpos_init, qvel=qvel_init)
            
            def step_fn(data, ctrl):
                data = data.replace(ctrl=ctrl)
                data = mjx.step(self.mjx_model, data)
                state_vec = jnp.concatenate([data.qpos, data.qvel])
                return data, state_vec
            
            _, states = jax.lax.scan(step_fn, data, traj_ctrl) # (function, initial value, sequence of inputs)

            init_state = jnp.concatenate([qpos_init, qvel_init])
            # prepend the initial state 
            states = jnp.concatenate([init_state[None], states], axis=0)
            return states
        
        # rollout all K samples 
        states = rollout_one_traj(controls)

        return states 
    
    @partial(jax.jit, static_argnums=0)
    def _mppi_step(
        self, 
        state: Float[Array, "d"], 
        U: Float[Array, "T a"], 
        key: PRNGKeyArray) -> Tuple[Float[Array, "T a"], Float[Array, "a"]]:
        # returns the updated control sequence and the next action to take

        # generate noise 
        key, noise_key = jax.random.split(key)
        noise = jax.random.normal(noise_key, (self.num_samples, self.horizon, self.act_dim)) * self.noise_sigma

        # perturbed controls 
        controls = U[None, :, :] + noise # K, T, A dim 

        # repeat the controls for frame skip 
        controls_repeated = jnp.repeat(controls, self.frame_skip, axis=1)

        # rollout the trajectories 
        rollout_states_repeated = self._rollout_trajectories(state, controls_repeated)
        rollout_states = rollout_states_repeated[:, ::self.frame_skip, :]

        # compute cost 
        costs = self.cost(rollout_states)

        # weighted costs with numerical stability
        beta = jnp.min(costs)
        exp_arg = -(costs - beta) / self.lambda_
        exp_costs = jnp.exp(exp_arg)
        weights = exp_costs / jnp.sum(exp_costs)

        # weighted update
        weighted_noise = jnp.einsum('k,kta->ta', weights, noise)
        updated_controls = U + weighted_noise

        # shift the control sequence 
        new_U = jnp.roll(updated_controls, -1, axis=0)
        new_U = new_U.at[-1].set(updated_controls[-1])
        
        action = updated_controls[0]
        return new_U, action
    
    def action(self, state: Float[np.ndarray, "d"]) -> Float[np.ndarray, "a"]:
        state_jax = jnp.asarray(state)
        self.key, subkey = jax.random.split(self.key)

        # run the MPPI step 
        new_U, action = self._mppi_step(state_jax, self.U, subkey)
        # update control seq import jax 
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
        num_samples: int = 16,
        horizon: int = 8,
        noise_sigma: float = 0.01,
        lambda_: float = 0.00001,
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
        
        # Pre-allocate data structure once; this way you don't initialize the mjdata at each step
        self.mjx_data_template = mjx.make_data(self.mjx_model) 
        
        # Pre-compute indices for state extraction
        self.nq = self.mjx_model.nq
        self.nv = self.mjx_model.nv
        self.state_dim = self.nq + self.nv

        init_control = jnp.zeros(self.act_dim)
        # tile repeats along the current dimensions 
        self.U = jnp.tile(init_control, (self.horizon, 1)) # self.horizon, 1 tells you how to repeat along each direction 

        # random key for JAX (needed for when you want to do JIT)
        self.key = jax.random.PRNGKey(0)

        self._compile_rollout()

    def _compile_rollout(self) -> None:
        dummy_state = jnp.zeros(self.state_dim)
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

        # initialize data 
        qpos_init = state[:self.nq]
        qvel_init = state[self.nq:]

        # vectorize over K trajectories
        @jax.vmap
        def rollout_one_traj(traj_ctrl):
            # traj_ctrl is control for one traj (T x act_dim)
            data = self.mjx_data_template.replace(qpos=qpos_init, qvel=qvel_init)
            
            def step_fn(data, ctrl):
                data = data.replace(ctrl=ctrl)
                data = mjx.step(self.mjx_model, data)
                state_vec = jnp.concatenate([data.qpos, data.qvel])
                return data, state_vec
            
            _, states = jax.lax.scan(step_fn, data, traj_ctrl) # (function, initial value, sequence of inputs)

            init_state = jnp.concatenate([qpos_init, qvel_init])
            # prepend the initial state 
            states = jnp.concatenate([init_state[None], states], axis=0)
            return states
        
        # rollout all K samples 
        states = rollout_one_traj(controls)

        return states 
    
    @partial(jax.jit, static_argnums=0)
    def _mppi_step(
        self, 
        state: Float[Array, "d"], 
        U: Float[Array, "T a"], 
        key: PRNGKeyArray) -> Tuple[Float[Array, "T a"], Float[Array, "a"]]:
        # returns the updated control sequence and the next action to take

        # generate noise 
        key, noise_key = jax.random.split(key)
        noise = jax.random.normal(noise_key, (self.num_samples, self.horizon, self.act_dim)) * self.noise_sigma

        # perturbed controls 
        controls = U[None, :, :] + noise # K, T, A dim 

        # repeat the controls for frame skip 
        controls_repeated = jnp.repeat(controls, self.frame_skip, axis=1)

        # rollout the trajectories 
        rollout_states_repeated = self._rollout_trajectories(state, controls_repeated)
        rollout_states = rollout_states_repeated[:, ::self.frame_skip, :]

        # compute cost 
        costs = self.cost(rollout_states)

        # weighted costs with numerical stability
        beta = jnp.min(costs)
        exp_arg = -(costs - beta) / self.lambda_
        exp_costs = jnp.exp(exp_arg)
        weights = exp_costs / jnp.sum(exp_costs)

        # weighted update
        weighted_noise = jnp.einsum('k,kta->ta', weights, noise)
        updated_controls = U + weighted_noise

        # shift the control sequence 
        new_U = jnp.roll(updated_controls, -1, axis=0)
        new_U = new_U.at[-1].set(updated_controls[-1])
        
        action = updated_controls[0]
        return new_U, action
    
    def action(self, state: Float[np.ndarray, "d"]) -> Float[np.ndarray, "a"]:
        state_jax = jnp.asarray(state)
        self.key, subkey = jax.random.split(self.key)

        # run the MPPI step 
        new_U, action = self._mppi_step(state_jax, self.U, subkey)
        # update control seq 
        self.U = new_U 
        action_np = np.asarray(action)

        return action_np
        self.U = new_U 
        action_np = np.asarray(action)

        return action_np