import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q_values = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.q_values(x)

    def get_q_values(self, x):
        q_values = self.q_values(x)
        return q_values

    def get_action(self, x):
        q_values = self.q_values(x)
        action = torch.argmax(q_values, dim=1)
        return action

class SAC_Actor(nn.Module):
    def __init__(self, env, state_dim, action_dim, logstd_min=-5, logstd_max=2):
        super().__init__()
        self.actor_public = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(256, action_dim),
        )
        self.actor_logstd = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.logstd_min = logstd_min
        self.logstd_max = logstd_max

    def forward(self, x):
        public_x = self.actor_public(x)
        mean = self.actor_mean(public_x)
        logstd = self.actor_logstd(public_x)
        logstd = self.logstd_min + 0.5 * (self.logstd_max - self.logstd_min) * (logstd + 1)
        return mean, logstd

    def get_action(self, x):
        mean, logstd = self(x)
        std = logstd.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class SAC_Critic(nn.Module):
    def __init__(self, env, state_dim, action_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.critic(x)
