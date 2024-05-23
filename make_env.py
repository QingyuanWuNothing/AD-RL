import gymnasium as gym
import numpy as np
from rich import print
from collections import deque

class ConstantObservationDelay(gym.Wrapper):
    def __init__(self, env, max_obs_delay_step=5):
        super().__init__(env)
        self.max_obs_delay_step = max_obs_delay_step
        self.obs_delay_buffer = deque(maxlen=max_obs_delay_step+1)
        self.obs_delay_step = 0
    
    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.obs_delay_buffer = deque(maxlen=self.max_obs_delay_step+1)
        self.obs_delay_buffer.append(observation)
        self.obs_delay_step = 0
        info['obs_delay_step'] = self.obs_delay_step
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self.obs_delay_buffer.append(observation)
        observation = self.obs_delay_buffer[0]
        if self.obs_delay_step < self.max_obs_delay_step:
            self.obs_delay_step += 1
        info['obs_delay_step'] = self.obs_delay_step

        return (
            observation, 
            reward, 
            terminated, 
            truncated, 
            info
        )

class ConstantActionDelay(gym.Wrapper):
    def __init__(self, env, max_action_delay_step=5):
        super().__init__(env)
        self.max_action_delay_step = max_action_delay_step
        self.action_delay_buffer = deque(maxlen=max_action_delay_step)
        self.action_delay_step = 0
    
    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.action_delay_buffer = deque(maxlen=self.max_action_delay_step)
        self.action_delay_step = 0
        info['action_delay_step'] = self.action_delay_step
        return observation, info

    def step(self, action):
        self.action_delay_buffer.append(action)
        action = self.action_delay_buffer[0]
        observation, reward, terminated, truncated, info = super().step(action)
        if self.action_delay_step < self.max_action_delay_step:
            self.action_delay_step += 1
        info['action_delay_step'] = self.action_delay_step

        return (
            observation, 
            reward, 
            terminated, 
            truncated, 
            info
        )

def make_classical_env(env_id, seed=0):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk

def make_vector_classical_envs(env_id, num_envs, seed):
    envs = gym.vector.SyncVectorEnv([make_classical_env(env_id, seed=i+seed) for i in range(num_envs)])
    return envs

def make_classical_obs_delay_env(env_id, seed=0, max_obs_delay_step=5):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = ConstantObservationDelay(env, max_obs_delay_step)
        env.action_space.seed(seed)
        return env
    return thunk

def make_vector_classical_obs_delay_envs(env_id, num_envs, seed, max_obs_delay_step=5):
    envs = gym.vector.SyncVectorEnv([make_classical_obs_delay_env(env_id, seed=seed+i, max_obs_delay_step=max_obs_delay_step) for i in range(num_envs)])
    return envs

def make_classical_action_delay_env(env_id, seed=0, max_action_delay_step=5):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = ConstantActionDelay(env, max_action_delay_step)
        env.action_space.seed(seed)
        return env
    return thunk

def make_vector_classical_action_delay_envs(env_id, num_envs, max_action_delay_step=5):
    envs = gym.vector.SyncVectorEnv([make_classical_action_delay_env(env_id, seed=i, max_action_delay_step=max_action_delay_step) for i in range(num_envs)])
    return envs

def make_mujoco_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=0.99)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def make_vector_mujoco_envs(env_id, num_envs):
    envs = gym.vector.SyncVectorEnv([make_mujoco_env(env_id, i) for i in range(num_envs)])
    return envs


def make_sac_mujoco_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def make_vector_sac_mujoco_envs(env_id, num_envs, seed):
    envs = gym.vector.SyncVectorEnv([make_sac_mujoco_env(env_id, seed+i) for i in range(num_envs)])
    return envs

def make_sac_mujoco_obs_delay_env(env_id, seed=0, max_obs_delay_step=5):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = ConstantObservationDelay(env, max_obs_delay_step)
        env.action_space.seed(seed)
        return env
    return thunk

def make_vector_sac_mujoco_obs_delay_envs(env_id, num_envs, seed, max_obs_delay_step=5):
    envs = gym.vector.SyncVectorEnv([make_sac_mujoco_obs_delay_env(env_id, seed+i, max_obs_delay_step) for i in range(num_envs)])
    return envs
