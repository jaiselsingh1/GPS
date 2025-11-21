import jax 
import jax.numpy as jnp 
import mujoco 
import mujoco.mjx as mjx 
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from jaxtyping import Float, PRNGKeyArray
from typing import Callable, Tuple
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
            data = mjx.step(self.mjx_model, data)
            # including time for now 
            state = jnp.concatenate([jnp.array([data.time]), data.qpos, data.qvel])
            return data, state
        
        # vectorize over K trajectories 
        def rollout_one_traj(data, traj_ctrl):
            # traj_ctrl is control for one traj (T x act_dim)
            _, states = jax.lax.scan(step_fn, data, traj_ctrl) # (function, initial value, sequence of inputs)

            init_state = jnp.concatenate([
                jnp.array([data.time]),
                data.qpos,
                data.qvel
            ])
            # prepend the initial state 
            states = jnp.concatenate([init_state[None], states], axis=0)
            return states
        
        # rollout all K samples 
        states = jax.vmap(rollout_one_traj)(data_batch, controls)

        return states 
    
    @jax.jit
    def _mppi_step(
        self, 
        state: Float[jnp.array, "d"], 
        U: Float[jnp.array, "T a"], 
        key: PRNGKeyArray) -> Tuple[Float[jnp.array, "T a"], Float[jnp.array, "a"]]:
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
        weighted_noise = jnp.sum(noise * weights[:, None, None], axis=0)
        updated_controls = U + weighted_noise

        # shift the control sequence 
        new_U = jnp.concatenate([
            updated_controls[1:, :], # drop the first control
            updated_controls[-1:, :]], # repeat the last control (the [-1:, :] is to get the last row but keep it as it's original dim)
            axis=0)
        
        action = updated_controls[0]
        return new_U, action
    
    def action(self, state: Float[jnp.array, "d"]) -> Float[jnp.array, "a"]:
        state_jax = jnp.array(state)
        self.key, subkey = jax.random.split(self.key)

        # run the MPPI step 
        new_U, action = self._mppi_step(state_jax, self.U, subkey)
        # update control seq 
        self.U = new_U 
        action = np.array(action)

        return action 





        






        
        















        

        

    


        
