import numpy as np
import random
from collections import deque
from tqdm import trange
import seaborn as sns
import matplotlib.pyplot as plt

GAMMA = 1.0
ALPHA = 0.2
EPSILON = 0.1
HORIZON = 100
STATE_INITIAL = (0, 0)
STATE_A = (HORIZON, 0)
STATE_B = (HORIZON, 1)
ACTION_A = 0
ACTION_B = 1
NUM_EPISODE = 100
NUM_SEED = 100

def reset():
    return STATE_INITIAL, False

def step(STATE, ACTION):
    TIMESTEP = STATE[0]
    REWARD = 0
    DONE = False
    NXT_TIMESTEP = TIMESTEP + 1
    NXT_STATE = (NXT_TIMESTEP, ACTION)
    if NXT_STATE == STATE_A:
        REWARD = 1
        DONE = True
    elif NXT_STATE == STATE_B:
        REWARD = 0
        DONE = True
    return REWARD, NXT_STATE, DONE

def sample_action():
    if random.random() < 0.5:
        return ACTION_A
    else:
        return ACTION_B

def greedy_action(Q_VALUE, STATE):
    if (STATE, ACTION_A) not in Q_VALUE.keys():
        Q_VALUE.update({(STATE, ACTION_A): 0}) 
    if (STATE, ACTION_B) not in Q_VALUE.keys():
        Q_VALUE.update({(STATE, ACTION_B): 0}) 
    
    if Q_VALUE[(STATE, ACTION_A)] > Q_VALUE[(STATE, ACTION_B)]:
        return ACTION_A
    else:
        return ACTION_B

def action_value(Q_VALUE, STATE, ACTION):
    if (STATE, ACTION_A) not in Q_VALUE.keys():
        Q_VALUE.update({(STATE, ACTION_A): 0}) 
    if (STATE, ACTION_B) not in Q_VALUE.keys():
        Q_VALUE.update({(STATE, ACTION_B): 0}) 
    return Q_VALUE[(STATE, ACTION)]

def update(Q_VALUE, STATE, ACTION, REWARD, NXT_STAET, DONE):
    if (STATE, ACTION) not in Q_VALUE.keys():
        Q_VALUE.update({(STATE, ACTION): 0}) 
    if (NXT_STAET, ACTION_A) not in Q_VALUE.keys():
        Q_VALUE.update({(NXT_STAET, ACTION_A): 0}) 
    if (NXT_STAET, ACTION_B) not in Q_VALUE.keys():
        Q_VALUE.update({(NXT_STAET, ACTION_B): 0}) 

    MAX_NXT_Q_VALUE = max(Q_VALUE[(NXT_STAET, ACTION_A)], 
                          Q_VALUE[(NXT_STAET, ACTION_B)])
    TD_TARGET = REWARD + GAMMA * DONE * MAX_NXT_Q_VALUE

    Q_VALUE[(STATE, ACTION)] += ALPHA * (TD_TARGET - Q_VALUE[(STATE, ACTION)])


def update_ad(AUX_Q_VALUE, AUG_Q_VALUE, 
              AUX_STATE, AUG_STATE, 
              ACTION, 
              REWARD, 
              NXT_AUX_STATE, NXT_AUG_STATE, 
              DONE):
    if (AUX_STATE, ACTION) not in AUX_Q_VALUE.keys():
        AUX_Q_VALUE.update({(AUX_STATE, ACTION): 0}) 
    if (NXT_AUX_STATE, ACTION_A) not in AUX_Q_VALUE.keys():
        AUX_Q_VALUE.update({(NXT_AUX_STATE, ACTION_A): 0}) 
    if (NXT_AUX_STATE, ACTION_B) not in AUX_Q_VALUE.keys():
        AUX_Q_VALUE.update({(NXT_AUX_STATE, ACTION_B): 0}) 

    if (AUG_STATE, ACTION) not in AUG_Q_VALUE.keys():
        AUG_Q_VALUE.update({(AUG_STATE, ACTION): 0}) 
    if (NXT_AUG_STATE, ACTION_A) not in AUG_Q_VALUE.keys():
        AUG_Q_VALUE.update({(NXT_AUG_STATE, ACTION_A): 0}) 
    if (NXT_AUG_STATE, ACTION_B) not in AUG_Q_VALUE.keys():
        AUG_Q_VALUE.update({(NXT_AUG_STATE, ACTION_B): 0}) 

    if AUG_Q_VALUE[(NXT_AUG_STATE, ACTION_A)] > AUG_Q_VALUE[(NXT_AUG_STATE, ACTION_A)]:
        MAX_NXT_Q_VALUE = AUX_Q_VALUE[(NXT_AUX_STATE, ACTION_A)]
    else:
        MAX_NXT_Q_VALUE = AUX_Q_VALUE[(NXT_AUX_STATE, ACTION_A)]

    TD_TARGET = REWARD + GAMMA * DONE * MAX_NXT_Q_VALUE

    AUG_Q_VALUE[(AUG_STATE, ACTION)] += ALPHA * (TD_TARGET - AUG_Q_VALUE[(AUG_STATE, ACTION)])


def rollout(Q_VALUE):
    eval_return = []
    for _ in range(1, 11):
        state, done = reset()
        episode_return = 0
        states_buffer = deque(maxlen=DELAY_STEP + 1)
        states_buffer.append(state)
        actions_buffer = deque(maxlen=DELAY_STEP)
        for _ in range(DELAY_STEP):
            states_buffer.append(state)
            actions_buffer.append(sample_action())
        aug_state = (states_buffer[0], tuple(actions_buffer))
        while done is False:
            action = greedy_action(Q_VALUE, aug_state)
            reward, state, done = step(state, action)
            states_buffer.append(state)
            actions_buffer.append(action)
            nxt_aug_state = (states_buffer[0], tuple(actions_buffer))
            episode_return += reward
            aug_state = nxt_aug_state
        eval_return.append(episode_return)
    return np.mean(eval_return), np.std(eval_return)


def make_aql_run():
    eval_step = []
    eval_mean = []

    for seed in trange(NUM_SEED):
        random.seed(seed)
        aug_q_value = dict()

        for episode in range(1, NUM_EPISODE + 1):
            eval_step.append(episode)
            state, done = reset()
            episode_return = 0
            states_buffer = deque(maxlen=DELAY_STEP + 1)
            states_buffer.append(state)
            actions_buffer = deque(maxlen=DELAY_STEP)
            for _ in range(DELAY_STEP):
                actions_buffer.append(sample_action())

            aug_state = (states_buffer[0], tuple(actions_buffer))
            while done is False:
                if random.random() < EPSILON:
                    action = sample_action()
                else:
                    action = greedy_action(aug_q_value, aug_state)
                reward, state, done = step(state, action)
                states_buffer.append(state)
                actions_buffer.append(action)
                
                nxt_aug_state = (states_buffer[0], tuple(actions_buffer))

                update(aug_q_value, aug_state, action, reward, nxt_aug_state, done)
                episode_return += reward
                aug_state = nxt_aug_state

            mean, _ = rollout(aug_q_value)
            eval_mean.append(mean)
    sns.lineplot(x=np.array(eval_step), y=np.array(eval_mean), label=r"A-QL")

def make_adql_run():
    eval_step = []
    eval_mean = []

    for seed in trange(NUM_SEED):
        random.seed(seed)
        aug_q_value = dict()
        aux_q_value = dict()

        for episode in range(1, NUM_EPISODE + 1):
            timestep = 0
            eval_step.append(episode)
            state, done = reset()
            episode_return = 0
            states_buffer = deque(maxlen=DELAY_STEP + 1)
            states_buffer.append(state)
            actions_buffer = deque(maxlen=DELAY_STEP)
            for _ in range(DELAY_STEP):
                states_buffer.append(state)
                actions_buffer.append(sample_action())

            aux_state = (states_buffer[DELAY_STEP - AUX_DELAY_STEP], tuple(list(actions_buffer)[DELAY_STEP - AUX_DELAY_STEP:]))
            aug_state = (states_buffer[0], tuple(actions_buffer))
            while done is False:
                if random.random() < EPSILON:
                    action = sample_action()
                else:
                    aux_action = greedy_action(aux_q_value, aux_state)
                    aug_action = greedy_action(aug_q_value, aug_state)
                    if action_value(aux_q_value, aux_state, aux_action) > action_value(aux_q_value, aux_state, aug_action):
                        action = aux_action
                    else:
                        action = aug_action

                reward, state, done = step(state, action)
                timestep += 1
                states_buffer.append(state)
                actions_buffer.append(action)
                
                nxt_aux_state = (states_buffer[DELAY_STEP - AUX_DELAY_STEP], tuple(list(actions_buffer)[DELAY_STEP - AUX_DELAY_STEP:]))
                nxt_aug_state = (states_buffer[0], tuple(actions_buffer))

                update(aux_q_value, aux_state, action, reward, nxt_aux_state, done)
                update_ad(aux_q_value, aug_q_value, 
                          aux_state, aug_state, 
                          action, 
                          reward, 
                          nxt_aux_state, nxt_aug_state, 
                          done)

                episode_return += reward
                aux_state = nxt_aux_state
                aug_state = nxt_aug_state

            mean, _ = rollout(aug_q_value)
            eval_mean.append(mean)
    sns.lineplot(x=np.array(eval_step), y=np.array(eval_mean), label=rf"AD-QL({AUX_DELAY_STEP}) (ours)")


if __name__ == '__main__':
    plt.figure(figsize=(8, 6))

    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
        '#8da0cb', 
        '#e78ac3', 
        '#a6d854',
        ])


    DELAY_STEP = 10
    make_aql_run()
    AUX_DELAY_STEP = 0
    make_adql_run()
    AUX_DELAY_STEP = 5
    make_adql_run()

    plt.grid(True, ls='--')
    plt.subplots_adjust(top=0.9, bottom=0.17, left=0.19, right=0.97)
    plt.legend(fontsize=20)
    plt.xlabel('Episodes', fontsize=30)
    plt.ylabel(r'Return ($\Delta=10$)', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.title(r'AD-QL(0) > AD-QL(5) > A-QL', fontsize=30)
    plt.savefig("deterministic_mdp.pdf")
    plt.close()
