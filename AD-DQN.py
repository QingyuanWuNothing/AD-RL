import os
import random
import time
from collections import deque
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from rich import print
from make_env import make_vector_classical_envs, make_vector_classical_obs_delay_envs
from nn import QNetwork
from utils import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--EXP_NAME", type=str, default=os.path.basename(__file__).rstrip(".py"))
parser.add_argument("--SEED", type=int, default=2024)
parser.add_argument("--EVAL_SEED", type=int, default=2023)
parser.add_argument("--TORCH_DETERMINISTIC", type=bool, default=True)
parser.add_argument("--DEVICE", default=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
parser.add_argument("--ENV_NAME", type=str, default="Acrobot-v1")
parser.add_argument("--STOCHASTIC", type=bool, default=False)
parser.add_argument("--NUM_ENVS", type=int, default=1)
parser.add_argument("--TOTAL_TIMESTEPS", type=int, default=int(5e5))
parser.add_argument("--GAMMA", type=float, default=0.99)
parser.add_argument("--DELAY_STEPS", type=int, default=int(10))
parser.add_argument("--BUFFER_SIZE", type=int, default=int(1e4))
parser.add_argument("--BATCH_SIZE", type=int, default=int(128))
parser.add_argument("--LR", type=float, default=2.5e-4)
parser.add_argument("--ANNEAL_LR", type=bool, default=True)
parser.add_argument("--START_EPSILON", type=int, default=1)
parser.add_argument("--FINAL_EPSILON", type=float, default=0.05)
parser.add_argument("--EXPLORATION_STEPS", type=int, default=int(2e5))
parser.add_argument("--LEARNING_START", type=int, default=int(1e4))
parser.add_argument("--TRAIN_FREQ", type=int, default=int(5))
parser.add_argument("--TARGET_NET_FREQ", type=int, default=int(5e2))
parser.add_argument("--TARGET_UPDATE_FACTOR", type=int, default=1)
parser.add_argument("--EVAL_NUMS", type=int, default=int(10))
parser.add_argument("--AUX_DELAY_STEPS", type=int, default=int(0))

class ReplayBuffer():
    def __init__(self, envs, aug_state_dim, aux_state_dim, buffer_size, env_nums):
        super().__init__()
        self.buffer = {
            'aug_states': torch.zeros((buffer_size, env_nums, aug_state_dim), dtype=torch.float32),
            'aux_states': torch.zeros((buffer_size, env_nums, aux_state_dim), dtype=torch.float32),
            'actions': torch.zeros((buffer_size, env_nums) + envs.single_action_space.shape, dtype=torch.int64),
            'rewards': torch.zeros(buffer_size, env_nums, dtype=torch.float32),
            'nxt_aug_states': torch.zeros((buffer_size, env_nums, aug_state_dim), dtype=torch.float32),
            'nxt_aux_states': torch.zeros((buffer_size, env_nums, aux_state_dim), dtype=torch.float32),
            'dones': torch.zeros(buffer_size, env_nums, dtype=torch.float32),
        }

        for key in self.buffer.keys():
            print(f"{key}: {self.buffer[key].shape}")

        self.buffer_size = buffer_size
        self.buffer_len = 0
        self.buffer_ptr = 0

    def store(self, aug_state, aux_state, action, reward, nxt_aug_state, nxt_aux_state, done):
        self.buffer['aug_states'][self.buffer_ptr] = torch.FloatTensor(aug_state)
        self.buffer['aux_states'][self.buffer_ptr] = torch.FloatTensor(aux_state)
        self.buffer['actions'][self.buffer_ptr] = torch.IntTensor(action)
        self.buffer['rewards'][self.buffer_ptr] = torch.FloatTensor(reward)
        self.buffer['nxt_aug_states'][self.buffer_ptr] = torch.FloatTensor(nxt_aug_state)
        self.buffer['nxt_aux_states'][self.buffer_ptr] = torch.FloatTensor(nxt_aux_state)
        self.buffer['dones'][self.buffer_ptr] = torch.LongTensor(done)

        self.buffer_ptr += 1
        if self.buffer_ptr >= self.buffer_size:
            self.buffer_ptr = 0

        if self.buffer_len < self.buffer_size:
            self.buffer_len += 1

    def sample(self, batch_size, device):
        indices = np.random.choice(self.buffer_len, size=batch_size)
        b_aug_states = torch.flatten(self.buffer['aug_states'][indices], 0, 1).to(device)
        b_aux_states = torch.flatten(self.buffer['aux_states'][indices], 0, 1).to(device)
        b_actions = torch.flatten(self.buffer['actions'][indices], 0, 1).to(device)
        b_rewards = torch.flatten(self.buffer['rewards'][indices], 0, 1).to(device)
        b_nxt_aug_states = torch.flatten(self.buffer['nxt_aug_states'][indices], 0, 1).to(device)
        b_nxt_aux_states = torch.flatten(self.buffer['nxt_aux_states'][indices], 0, 1).to(device)
        b_dones = torch.flatten(self.buffer['dones'][indices], 0, 1).to(device)
        return b_aug_states, b_aux_states, b_actions, b_rewards, b_nxt_aug_states, b_nxt_aux_states, b_dones

def make_train(config):
    print(config)

    file_name = config["EXP_NAME"]
    if not os.path.exists(f"runs/{file_name}"):
        os.makedirs(f"runs/{file_name}")

    exp_tag = f'DELAY_STEPS={config["DELAY_STEPS"]}/AUX_DELAY_STEPS={config["AUX_DELAY_STEPS"]}/{config["ENV_NAME"]}_SEED_{config["SEED"]}_{int(time.time())}'

    logger = SummaryWriter(f"runs/{file_name}/{exp_tag}")
    logger.add_text(
        "config",
        "|parametrix|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
    )

    make_seeding(config["SEED"])

    envs = make_vector_classical_envs(env_id=config["ENV_NAME"], 
                                      num_envs=config["NUM_ENVS"],
                                      seed=config["SEED"])
    obs, _ = envs.reset()
    state = torch.tensor(obs)

    state_dim = state.shape[1]
    action_dim = envs.single_action_space.n

    states_buffer = deque(maxlen=config["DELAY_STEPS"] + 1)
    states_buffer.append(state)
    actions_buffer = deque(maxlen=config["DELAY_STEPS"])
    for _ in range(config["DELAY_STEPS"]):
        actions_buffer.append(envs.action_space.sample())

    aug_state = torch.cat((states_buffer[0], 
                           torch.tensor(np.array(actions_buffer)).transpose(0, -1)), dim=-1)
    aug_state_dim = aug_state.shape[1]

    aux_state = torch.cat((states_buffer[0], 
                               torch.tensor(np.array(actions_buffer)[config["DELAY_STEPS"]-config["AUX_DELAY_STEPS"]:]).transpose(0, -1)), dim=-1)
    aux_state_dim = aux_state.shape[1]

    aug_q_network = QNetwork(aug_state_dim, action_dim).to(config["DEVICE"])
    aug_optimizer = optim.Adam(aug_q_network.parameters(), lr=config["LR"], eps=1e-5)
    aug_q_target = QNetwork(aug_state_dim, action_dim).to(config["DEVICE"])
    aug_q_target.load_state_dict(aug_q_network.state_dict())

    aux_q_network = QNetwork(aux_state_dim, action_dim).to(config["DEVICE"])
    aux_optimizer = optim.Adam(aux_q_network.parameters(), lr=config["LR"], eps=1e-5)
    aux_q_target = QNetwork(aux_state_dim, action_dim).to(config["DEVICE"])
    aux_q_target.load_state_dict(aux_q_network.state_dict())

    replay_buffer = ReplayBuffer(envs, aug_state_dim, aux_state_dim, config["BUFFER_SIZE"], config["NUM_ENVS"])

    metric = {
        "moving_avg_return": deque(maxlen=50),
        "moving_avg_length": deque(maxlen=50),
        "best_avg_return": 0,
    }

    global_step_bar = trange(1, config["TOTAL_TIMESTEPS"] + 1, desc=f"runs/{file_name}/{exp_tag}")
    for global_step in global_step_bar:
        epsilon = make_linear_schedule(config["START_EPSILON"], config["FINAL_EPSILON"], config["EXPLORATION_STEPS"], global_step)

        if random.random() < epsilon:
            actions = envs.action_space.sample()
        else:
            with torch.no_grad():
                aug_actions = aug_q_network.get_action(aug_state)
                aux_actions = aux_q_network.get_action(aux_state)
                aug_q_val = aux_q_network(aux_state).gather(1, aug_actions.unsqueeze(-1)).squeeze()
                aux_q_val = aux_q_network(aux_state).gather(1, aux_actions.unsqueeze(-1)).squeeze()
                if aug_q_val < aux_q_val:
                    actions = aug_actions.cpu().numpy()
                else:
                    actions = aux_actions.cpu().numpy()

        if config["STOCHASTIC"] and random.random() < 0.1:
            next_obs, rewards, terminated, truncated, infos = envs.step(envs.action_space.sample())
        else:
            next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        dones = np.logical_or(terminated, truncated)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    metric["moving_avg_return"].append(info["episode"]["r"])
                    metric["moving_avg_length"].append(info["episode"]["l"])
            states_buffer = deque(maxlen=config["DELAY_STEPS"]+1)
            actions_buffer = deque(maxlen=config["DELAY_STEPS"])
            state = torch.tensor(next_obs)
            states_buffer.append(state)

            for _ in range(config["DELAY_STEPS"]):
                actions_buffer.append(envs.action_space.sample())

        state = torch.tensor(next_obs)
        states_buffer.append(state)
        actions_buffer.append(actions)

        nxt_aug_state = torch.cat((states_buffer[0], 
                                   torch.tensor(np.array(actions_buffer)).transpose(0, -1)), dim=-1)
    
        if config["DELAY_STEPS"]-config["AUX_DELAY_STEPS"] >= len(states_buffer):
            nxt_aux_state = torch.cat((states_buffer[-1], 
                                        torch.tensor(np.array(actions_buffer)[config["DELAY_STEPS"]-config["AUX_DELAY_STEPS"]:]).transpose(0, -1)), dim=-1)
        else:
            nxt_aux_state = torch.cat((states_buffer[config["DELAY_STEPS"]-config["AUX_DELAY_STEPS"]], 
                            torch.tensor(np.array(actions_buffer)[config["DELAY_STEPS"]-config["AUX_DELAY_STEPS"]:]).transpose(0, -1)), dim=-1)

        replay_buffer.store(aug_state, aux_state, actions, rewards, nxt_aug_state, nxt_aux_state, dones)

        aug_state = nxt_aug_state
        aux_state = nxt_aux_state

        if global_step >= config["LEARNING_START"]:
            if global_step % config["TRAIN_FREQ"] == 0:
                b_aug_states, b_aux_states, b_actions, b_rewards, b_nxt_aug_states, b_nxt_aux_states, b_dones = replay_buffer.sample(config["BATCH_SIZE"], config["DEVICE"])

                if config["ANNEAL_LR"]:
                    aux_optimizer.param_groups[0]["lr"] = make_anneal_lr(global_step, config["TOTAL_TIMESTEPS"] + 1, config["LR"])
                    aug_optimizer.param_groups[0]["lr"] = make_anneal_lr(global_step, config["TOTAL_TIMESTEPS"] + 1, config["LR"])
                if random.random() < 0.5:
                    with torch.no_grad():
                        aux_target_max, _ = aux_q_target(b_nxt_aux_states).max(dim=1)
                        b_aux_td_target = b_rewards + config["GAMMA"] * aux_target_max * (1 - b_dones)
                    b_aux_q_val = aux_q_network(b_aux_states).gather(1, b_actions.unsqueeze(-1)).squeeze()
                    aux_loss = F.mse_loss(b_aux_td_target, b_aux_q_val)
                    aux_optimizer.zero_grad()
                    aux_loss.backward()
                    aux_optimizer.step()
                else:
                    with torch.no_grad():
                        _, b_nxt_actions = aug_q_target(b_nxt_aug_states).max(dim=1)
                        aux_target = torch.gather(aux_q_target(b_nxt_aux_states), 1, b_nxt_actions.unsqueeze(-1)).squeeze(-1)
                        b_td_target = b_rewards + config["GAMMA"] * aux_target * (1 - b_dones)
                    b_q_val = aug_q_network(b_aug_states).gather(1, b_actions.unsqueeze(-1)).squeeze()
                    loss = F.mse_loss(b_td_target, b_q_val)
                    aug_optimizer.zero_grad()
                    loss.backward()
                    aug_optimizer.step()

            if global_step % config["TARGET_NET_FREQ"] == 0:
                for target_network_param, q_network_param in zip(aug_q_target.parameters(), aug_q_network.parameters()):
                    target_network_param.data.copy_(config["TARGET_UPDATE_FACTOR"] * q_network_param.data + (1.0 - config["TARGET_UPDATE_FACTOR"]) * target_network_param.data)
                for target_network_param, q_network_param in zip(aux_q_target.parameters(), aux_q_network.parameters()):
                    target_network_param.data.copy_(config["TARGET_UPDATE_FACTOR"] * q_network_param.data + (1.0 - config["TARGET_UPDATE_FACTOR"]) * target_network_param.data)
    
        if global_step % 10000 == 0:
            global_step_bar.set_postfix(global_step=global_step, avg_return_mean=np.mean(metric["moving_avg_return"]))
            logger.add_scalar("performace/moving_avg_return_mean", np.mean(metric["moving_avg_return"]), global_step)
            logger.add_scalar("performace/moving_avg_return_std", np.std(metric["moving_avg_return"]), global_step)
            logger.add_scalar("performace/moving_avg_length_mean", np.mean(metric["moving_avg_length"]), global_step)
            logger.add_scalar("performace/moving_avg_length_std", np.std(metric["moving_avg_length"]), global_step)
            eval_return_mean = make_eval(aug_q_network, config)
            logger.add_scalar("performace/eval_return_mean", eval_return_mean, global_step)

def make_eval(aug_q_network, config):
    metric = {
        "eval_num": int(0),
        "eval_return": deque(maxlen=config["EVAL_NUMS"]),
        "eval_length": deque(maxlen=config["EVAL_NUMS"])
    }
    envs = make_vector_classical_obs_delay_envs(env_id=config["ENV_NAME"], 
                                                num_envs=config["NUM_ENVS"], 
                                                seed=config["EVAL_SEED"], 
                                                max_obs_delay_step=config["DELAY_STEPS"])

    obs, infos = envs.reset()
    obs = torch.Tensor(obs).to(config["DEVICE"])
    actions_buffer = deque(maxlen=config["DELAY_STEPS"])
    for _ in range(config["DELAY_STEPS"]):
        actions_buffer.append(envs.action_space.sample())
    aug_state = torch.cat((obs, torch.tensor(np.array(actions_buffer)).transpose(0, -1)), dim=-1)

    while metric["eval_num"] < config["EVAL_NUMS"]:
        with torch.no_grad():
            action = aug_q_network.get_action(aug_state)
        if config["STOCHASTIC"] and random.random() < 0.1:
            next_obs, _, _, _, infos = envs.step(envs.action_space.sample())
        else:
            next_obs, _, _, _, infos = envs.step(action.cpu().numpy())
        obs = torch.Tensor(next_obs).to(config["DEVICE"])
        actions_buffer.append(action.cpu().numpy())    
        aug_action_states = torch.tensor(np.array(actions_buffer)).transpose(0, -1).to(config["DEVICE"])
        aug_state = torch.cat((obs, aug_action_states), dim=-1)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    metric["eval_num"] += 1
                    metric["eval_return"].append(info["episode"]["r"])
                    metric["eval_length"].append(info["episode"]["l"])
            action_buffer = deque(maxlen=config["DELAY_STEPS"])
            for _ in range(config["DELAY_STEPS"]):
                action_buffer.append(envs.action_space.sample())

    print(f'eval_return {np.array(metric["eval_return"]).mean()}')
    return np.array(metric["eval_return"]).mean()

if __name__ == '__main__':
    config = vars(parser.parse_args())
    make_train(config)