import os
import time
from collections import deque
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from rich import print
from make_env import make_vector_sac_mujoco_envs, make_vector_sac_mujoco_obs_delay_envs
from nn import SAC_Actor, SAC_Critic
from utils import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--EXP_NAME", type=str, default=os.path.basename(__file__).rstrip(".py"))
parser.add_argument("--SEED", type=int, default=2024)
parser.add_argument("--EVAL_SEED", type=int, default=2025)
parser.add_argument("--TORCH_DETERMINISTIC", type=bool, default=True)
parser.add_argument("--DEVICE", default=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
parser.add_argument("--ENV_NAME", type=str, default="Ant-v4")
parser.add_argument("--NUM_ENVS", type=int, default=1)
parser.add_argument("--TOTAL_TIMESTEPS", type=int, default=int(1e6))
parser.add_argument("--GAMMA", type=float, default=0.99)
parser.add_argument("--DELAY_STEPS", type=int, default=int(25))
parser.add_argument("--BUFFER_SIZE", type=int, default=int(1e6))
parser.add_argument("--BATCH_SIZE", type=int, default=int(256))
parser.add_argument("--ACTOR_LR", type=float, default=3e-4)
parser.add_argument("--CRITIC_LR", type=float, default=1e-3)
parser.add_argument("--LEARNING_START", type=int, default=int(5e3))
parser.add_argument("--POLICY_FREQ", type=int, default=int(2))
parser.add_argument("--TARGET_FREQ", type=int, default=int(1))
parser.add_argument("--ALPHA", type=float, default=0.2)
parser.add_argument("--AUTO_TUNE", type=bool, default=True)
parser.add_argument("--ALPHA_LR", type=float, default=1e-3)
parser.add_argument("--TARGET_UPDATE_FACTOR", type=float, default=5e-3)
parser.add_argument("--EVAL_NUMS", type=int, default=int(10))
parser.add_argument("--N_STEPS", type=int, default=int(3))
parser.add_argument("--AUX_DELAY_STEPS", type=int, default=int(0))


class ReplayBuffer():
    def __init__(self, envs, aug_state_dim, aux_state_dim, buffer_size, env_nums):
        super().__init__()
        self.buffer = {
            'aug_states': torch.zeros((buffer_size, env_nums, aug_state_dim), dtype=torch.float32),
            'aux_states': torch.zeros((buffer_size, env_nums, aux_state_dim), dtype=torch.float32),
            'actions': torch.zeros((buffer_size, env_nums) + envs.single_action_space.shape, dtype=torch.float32),
            'rewards': torch.zeros(buffer_size, env_nums, dtype=torch.float32),
            'nxt_aug_states': torch.zeros((buffer_size, env_nums, aug_state_dim), dtype=torch.float32),
            'nxt_aux_states': torch.zeros((buffer_size, env_nums, aux_state_dim), dtype=torch.float32),
            'dones': torch.zeros(buffer_size, env_nums, dtype=torch.float32),
            'discount_factors': torch.zeros(buffer_size, env_nums, dtype=torch.float32),
        }

        for key in self.buffer.keys():
            print(f"{key}: {self.buffer[key].shape}")

        self.buffer_size = buffer_size
        self.buffer_len = 0
        self.buffer_ptr = 0

    def store(self, aug_state, aux_state, action, reward, nxt_aug_state, nxt_aux_state, done, discount_factor):
        self.buffer['aug_states'][self.buffer_ptr] = torch.FloatTensor(aug_state)
        self.buffer['aux_states'][self.buffer_ptr] = torch.FloatTensor(aux_state)
        self.buffer['actions'][self.buffer_ptr] = torch.FloatTensor(action)
        self.buffer['rewards'][self.buffer_ptr] = torch.FloatTensor(reward)
        self.buffer['nxt_aug_states'][self.buffer_ptr] = torch.FloatTensor(nxt_aug_state)
        self.buffer['nxt_aux_states'][self.buffer_ptr] = torch.FloatTensor(nxt_aux_state)
        self.buffer['dones'][self.buffer_ptr] = torch.LongTensor(done)
        self.buffer['discount_factors'][self.buffer_ptr] = torch.FloatTensor(discount_factor)

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
        b_discount_factors = torch.flatten(self.buffer['discount_factors'][indices], 0, 1).to(device)
        return b_aug_states, b_aux_states, b_actions, b_rewards, b_nxt_aug_states, b_nxt_aux_states, b_dones, b_discount_factors

def make_train(config):
    print(config)

    file_name = config["EXP_NAME"].replace("n_step", f'{config["N_STEPS"]}_step')
    if not os.path.exists(f"runs/{file_name}"):
        os.makedirs(f"runs/{file_name}")

    exp_tag = f'DELAY_STEPS={config["DELAY_STEPS"]}/AUX_DELAY_STEPS={config["AUX_DELAY_STEPS"]}/{config["ENV_NAME"]}_SEED_{config["SEED"]}_{int(time.time())}'

    logger = SummaryWriter(f"runs/{file_name}/{exp_tag}")
    logger.add_text(
        "config",
        "|parametrix|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
    )

    make_seeding(config["SEED"])

    envs = make_vector_sac_mujoco_envs(env_id=config["ENV_NAME"], 
                                       num_envs=config["NUM_ENVS"],
                                       seed=config["SEED"])
    obs, _ = envs.reset()
    state = torch.tensor(obs).float()

    state_dim = state.shape[1]
    action_dim = envs.action_space.sample().shape[1]
    
    states_buffer = deque(maxlen=config["DELAY_STEPS"] + 1)
    states_buffer.append(state)
    actions_buffer = deque(maxlen=config["DELAY_STEPS"])
    for _ in range(config["DELAY_STEPS"]):
        actions_buffer.append(envs.action_space.sample())

    aug_state = torch.cat((states_buffer[0], 
                           torch.tensor(np.array(actions_buffer)).transpose(0, 1).view(config["NUM_ENVS"], -1)), dim=-1)
    aug_state_dim = aug_state.shape[1]
    
    aux_state = torch.cat((states_buffer[0], 
                            torch.tensor(np.array(actions_buffer)[config["DELAY_STEPS"]-config["AUX_DELAY_STEPS"]:]).transpose(0, -1).reshape(config["NUM_ENVS"], -1)), dim=-1)
    aux_state_dim = aux_state.shape[1]

    aug_actor = SAC_Actor(envs, aug_state_dim, action_dim).to(config["DEVICE"])
    aug_actor_optimizer = optim.Adam(list(aug_actor.parameters()), lr=config["ACTOR_LR"])

    aux_actor = SAC_Actor(envs, aux_state_dim, action_dim).to(config["DEVICE"])
    aux_actor_optimizer = optim.Adam(list(aux_actor.parameters()), lr=config["ACTOR_LR"])
    aux_critic_1 = SAC_Critic(envs, aux_state_dim, action_dim).to(config["DEVICE"])
    aux_target_1 = SAC_Critic(envs, aux_state_dim, action_dim).to(config["DEVICE"])
    aux_target_1.load_state_dict(aux_critic_1.state_dict())
    aux_critic_2 = SAC_Critic(envs, aux_state_dim, action_dim).to(config["DEVICE"])
    aux_target_2 = SAC_Critic(envs, aux_state_dim, action_dim).to(config["DEVICE"])
    aux_target_2.load_state_dict(aux_critic_2.state_dict())
    aux_critic_optimizer = optim.Adam(list(aux_critic_1.parameters()) + list(aux_critic_2.parameters()), lr=config["CRITIC_LR"])

    if config["AUTO_TUNE"]:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(config["DEVICE"])).item()

        log_aug_alpha = torch.zeros(1, requires_grad=True, device=config["DEVICE"])
        aug_alpha = log_aug_alpha.exp().item()
        aug_alpha_optimizer = optim.Adam([log_aug_alpha], lr=config["ALPHA_LR"])

        log_aux_alpha = torch.zeros(1, requires_grad=True, device=config["DEVICE"])
        aux_alpha = log_aux_alpha.exp().item()
        aux_alpha_optimizer = optim.Adam([log_aux_alpha], lr=config["ALPHA_LR"])
    else:
        aug_alpha = config["ALPHA"]
        aux_alpha = config["ALPHA"]

    replay_buffer = ReplayBuffer(envs, aug_state_dim, aux_state_dim, config["BUFFER_SIZE"], config["NUM_ENVS"])

    envs.single_observation_space.dtype = np.float32

    metric = {
        "moving_avg_return": deque(maxlen=50),
        "moving_avg_length": deque(maxlen=50),
        "best_avg_return": -np.inf,
    }

    global_step_bar = trange(1, config["TOTAL_TIMESTEPS"] + 1, desc=f"runs/{file_name}/{exp_tag}")
    rollout_n_step_buffer = {
        'aug_states': deque(maxlen=config["N_STEPS"] + 1),
        'aux_states': deque(maxlen=config["N_STEPS"] + 1),
        'actions': deque(maxlen=config["N_STEPS"]),
        'rewards': deque(maxlen=config["N_STEPS"]),
        'dones': deque(maxlen=config["N_STEPS"]),
    }
    rollout_n_step_buffer['aug_states'].append(aug_state)
    rollout_n_step_buffer['aux_states'].append(aux_state)

    for global_step in global_step_bar:
        if global_step < config["LEARNING_START"]:
            actions = envs.action_space.sample()
        else:
            with torch.no_grad():
                actions, _, _ = aux_actor.get_action(aux_state.to(config["DEVICE"]))
                critic_1_q_value = aux_critic_1(aux_state.to(config["DEVICE"]), actions)
                critic_2_q_value = aux_critic_2(aux_state.to(config["DEVICE"]), actions)
                q_value = torch.min(critic_1_q_value, critic_2_q_value)

                actions, _, _ = aug_actor.get_action(aug_state.to(config["DEVICE"]))
                critic_1_q_value = aux_critic_1(aux_state.to(config["DEVICE"]), actions)
                critic_2_q_value = aux_critic_2(aux_state.to(config["DEVICE"]), actions)
                aug_q_value = torch.min(critic_1_q_value, critic_2_q_value)
                if q_value < aug_q_value:
                    actions, _, _ = aux_actor.get_action(aux_state.to(config["DEVICE"]))
                else:
                    actions, _, _ = aug_actor.get_action(aug_state.to(config["DEVICE"]))
                actions = actions.detach().cpu().numpy()
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        dones = np.logical_or(terminated, truncated)


        state = torch.tensor(next_obs).float()
        states_buffer.append(state)
        actions_buffer.append(actions)
    
        nxt_aug_state = torch.cat((states_buffer[0], 
                                   torch.tensor(np.array(actions_buffer)).transpose(0, 1).view(config["NUM_ENVS"], -1)), dim=-1).float()

        nxt_aux_state = state
        if config["DELAY_STEPS"]-config["AUX_DELAY_STEPS"] >= len(states_buffer):
            nxt_aux_state = torch.cat((states_buffer[-1], 
                                       torch.tensor(np.array(actions_buffer)[config["DELAY_STEPS"]-config["AUX_DELAY_STEPS"]:]).transpose(0, -1).reshape(config["NUM_ENVS"], -1)), dim=-1).float()
        else:
            nxt_aux_state = torch.cat((states_buffer[config["DELAY_STEPS"]-config["AUX_DELAY_STEPS"]], 
                                       torch.tensor(np.array(actions_buffer)[config["DELAY_STEPS"]-config["AUX_DELAY_STEPS"]:]).transpose(0, -1).reshape(config["NUM_ENVS"], -1)), dim=-1).float()
        aug_state = nxt_aug_state
        aux_state = nxt_aux_state

        rollout_n_step_buffer['aug_states'].append(aug_state)
        rollout_n_step_buffer['aux_states'].append(aux_state)
        rollout_n_step_buffer['actions'].append(actions)
        rollout_n_step_buffer['rewards'].append(rewards)
        rollout_n_step_buffer['dones'].append(dones)

        if len(rollout_n_step_buffer['aug_states']) == config["N_STEPS"] + 1:
            n_step_returns = 0
            for i in range(config["N_STEPS"]):
                n_step_returns += pow(config['GAMMA'], i)*rollout_n_step_buffer['rewards'][i]
                replay_buffer.store(aug_state=rollout_n_step_buffer['aug_states'][0],
                                    aux_state=rollout_n_step_buffer['aux_states'][0],
                                    action=rollout_n_step_buffer['actions'][0],
                                    reward=n_step_returns,
                                    nxt_aug_state=rollout_n_step_buffer['aug_states'][i+1],
                                    nxt_aux_state=rollout_n_step_buffer['aux_states'][i+1],
                                    done=rollout_n_step_buffer['dones'][i],
                                    discount_factor=np.array([pow(config['GAMMA'], i+1)]))

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    metric["moving_avg_return"].append(info["episode"]["r"])
                    metric["moving_avg_length"].append(info["episode"]["l"])
            states_buffer = deque(maxlen=config["DELAY_STEPS"]+1)
            states_buffer.append(state)
            actions_buffer = deque(maxlen=config["DELAY_STEPS"])
            for _ in range(config["DELAY_STEPS"]):
                actions_buffer.append(envs.action_space.sample())
            rollout_n_step_buffer = {
                'aug_states': deque(maxlen=config["N_STEPS"] + 1),
                'aux_states': deque(maxlen=config["N_STEPS"] + 1),
                'actions': deque(maxlen=config["N_STEPS"]),
                'rewards': deque(maxlen=config["N_STEPS"]),
                'dones': deque(maxlen=config["N_STEPS"]),
            }
            rollout_n_step_buffer['aug_states'].append(aug_state)
            rollout_n_step_buffer['aux_states'].append(aux_state)

        if global_step > config["LEARNING_START"]:
            b_aug_states, b_aux_states, b_actions, b_rewards, b_nxt_aug_states, b_nxt_aux_states, b_dones, b_discount_factors = replay_buffer.sample(config["BATCH_SIZE"], config["DEVICE"])
            with torch.no_grad():
                nxt_state_actions, nxt_state_log_pi, _ = aux_actor.get_action(b_nxt_aux_states)
                target_1_nxt_value = aux_target_1(b_nxt_aux_states, nxt_state_actions) - aux_alpha * nxt_state_log_pi
                nxt_state_actions, nxt_state_log_pi, _ = aug_actor.get_action(b_nxt_aug_states)
                target_2_nxt_value = aux_target_2(b_nxt_aux_states, nxt_state_actions) - aug_alpha * nxt_state_log_pi
                min_target_nxt_value = torch.min(target_1_nxt_value, target_2_nxt_value)
                next_q_value = b_rewards.flatten() + (1 - b_dones.flatten()) * b_discount_factors * (min_target_nxt_value).view(-1)
                
            critic_1_values = aux_critic_1(b_aux_states, b_actions).view(-1)
            critic_2_values = aux_critic_2(b_aux_states, b_actions).view(-1)
            critic_1_loss = F.mse_loss(critic_1_values, next_q_value)
            critic_2_loss = F.mse_loss(critic_2_values, next_q_value)
            critic_loss = critic_1_loss + critic_2_loss

            aux_critic_optimizer.zero_grad()
            critic_loss.backward()
            aux_critic_optimizer.step()

            if global_step % config["POLICY_FREQ"] == 0:
                for _ in range(config["POLICY_FREQ"]):
                    if random.random() < 0.5:
                        actions, log_prob, _ = aux_actor.get_action(b_aux_states)
                        critic_1_q_value = aux_critic_1(b_aux_states, actions)
                        critic_2_q_value = aux_critic_2(b_aux_states, actions)
                        min_critic_q_value = torch.min(critic_1_q_value, critic_2_q_value)

                        actor_loss = ((aux_alpha * log_prob) - min_critic_q_value).mean()
                        aux_actor_optimizer.zero_grad()
                        actor_loss.backward()
                        aux_actor_optimizer.step()
                    else:
                        aug_action, aug_log_prob, _ = aug_actor.get_action(b_aug_states)
                        critic_1_q_value = aux_critic_1(b_aux_states, aug_action)
                        critic_2_q_value = aux_critic_1(b_aux_states, aug_action)
                        min_critic_q_value = torch.min(critic_1_q_value, critic_2_q_value)

                        aug_actor_loss = ((aug_alpha * aug_log_prob) - min_critic_q_value).mean()
                        aug_actor_optimizer.zero_grad()
                        aug_actor_loss.backward()
                        aug_actor_optimizer.step()

                    if config["AUTO_TUNE"]:
                        with torch.no_grad():
                            _, log_prob, _ = aux_actor.get_action(b_aux_states)
                        aux_alpha_loss = (-log_aux_alpha.exp() * (log_prob + target_entropy)).mean()

                        aux_alpha_optimizer.zero_grad()
                        aux_alpha_loss.backward()
                        aux_alpha_optimizer.step()
                        aux_alpha = log_aux_alpha.exp().item()

                        with torch.no_grad():
                            _, aug_log_prob, _ = aug_actor.get_action(b_aug_states)
                        aug_alpha_loss = (-log_aug_alpha.exp() * (aug_log_prob + target_entropy)).mean()

                        aug_alpha_optimizer.zero_grad()
                        aug_alpha_loss.backward()
                        aug_alpha_optimizer.step()
                        aug_alpha = log_aug_alpha.exp().item()

            if global_step % config["TARGET_FREQ"] == 0:
                for param, target_param in zip(aux_critic_1.parameters(), aux_target_1.parameters()):
                    target_param.data.copy_(config["TARGET_UPDATE_FACTOR"] * param.data + (1 - config["TARGET_UPDATE_FACTOR"]) * target_param.data)
                for param, target_param in zip(aux_critic_2.parameters(), aux_target_2.parameters()):
                    target_param.data.copy_(config["TARGET_UPDATE_FACTOR"] * param.data + (1 - config["TARGET_UPDATE_FACTOR"]) * target_param.data)

        if global_step % 10000 == 0:
            global_step_bar.set_postfix(global_step=global_step, avg_return_mean=np.mean(metric["moving_avg_return"]))
            logger.add_scalar("performace/moving_avg_return_mean", np.mean(metric["moving_avg_return"]), global_step)
            logger.add_scalar("performace/moving_avg_return_std", np.std(metric["moving_avg_return"]), global_step)
            logger.add_scalar("performace/moving_avg_length_mean", np.mean(metric["moving_avg_length"]), global_step)
            logger.add_scalar("performace/moving_avg_length_std", np.std(metric["moving_avg_length"]), global_step)
            eval_return_mean = make_eval(aug_actor, config)
            logger.add_scalar("performace/eval_return_mean", eval_return_mean, global_step)

def make_eval(aug_actor, config):
    metric = {
        "eval_num": int(0),
        "eval_return": deque(maxlen=config["EVAL_NUMS"]),
        "eval_length": deque(maxlen=config["EVAL_NUMS"])
    }
    envs = make_vector_sac_mujoco_obs_delay_envs(env_id=config["ENV_NAME"], 
                                                 num_envs=config["NUM_ENVS"], 
                                                 seed=config["EVAL_SEED"],
                                                 max_obs_delay_step=config["DELAY_STEPS"])

    obs, _ = envs.reset()
    state = torch.tensor(obs).float()
    actions_buffer = deque(maxlen=config["DELAY_STEPS"])
    for _ in range(config["DELAY_STEPS"]):
        actions_buffer.append(envs.action_space.sample())
    aug_state = torch.cat((state, torch.tensor(np.array(actions_buffer)).transpose(0, 1).view(config["NUM_ENVS"], -1)), dim=-1)

    while metric["eval_num"] < config["EVAL_NUMS"]:
        with torch.no_grad():
            actions, _, _ = aug_actor.get_action(aug_state.to(config["DEVICE"]))
        actions = actions.detach().cpu().numpy()
        next_obs, _, _, _, infos = envs.step(actions)
        actions_buffer.append(actions)
        aug_state = torch.cat((torch.tensor(next_obs), torch.tensor(np.array(actions_buffer)).transpose(0, 1).view(config["NUM_ENVS"], -1)), dim=-1).float()

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    metric["eval_num"] += 1
                    metric["eval_return"].append(info["episode"]["r"])
                    metric["eval_length"].append(info["episode"]["l"])
            actions_buffer = deque(maxlen=config["DELAY_STEPS"])
            for _ in range(config["DELAY_STEPS"]):
                actions_buffer.append(envs.action_space.sample())

    print(f'eval_return {np.array(metric["eval_return"]).mean()}')
    return np.array(metric["eval_return"]).mean()

if __name__ == '__main__':
    config = vars(parser.parse_args())
    make_train(config)