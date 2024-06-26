# AD-RL: Boosting Reinforcement Learning with Strongly Delayed Feedback Through Auxiliary Short Delays

## guidelines
### 1. requirement
#### 1.1 setup the environment
    conda create -n AD_RL python=3.10
    conda activate AD_RL
    pip install -r requirement.yaml
#### 1.2 install the benchmarks
    pip install gymnasium[mujoco]
### 2. run the code
#### 2.1 AD-QL (Deterministic MDP)
    python3 DETERMINISTIC-MDP.py
#### 2.2 AD-QL (Stochastic MDP)
    python3 STOCHASTIC-MDP.py
#### 2.3 AD-DQN (Determinisitc Acrobot)
    python3 AD-DQN.py --ENV_NAME=Acrobot-v1 --STOCHASTIC=False --DELAY_STEPS=10 --AUX_DELAY_STEPS=0
#### 2.4 AD-DQN (Stochastic Acrobot)
    python3 AD-DQN.py --ENV_NAME=Acrobot-v1 --STOCHASTIC=True --DELAY_STEPS=20 --AUX_DELAY_STEPS=1
#### 2.5 AD-SAC (MuJoCo)
    python3 AD-SAC.py --ENV_NAME=Ant-v4 --DELAY_STEPS=25 --AUX_DELAY_STEPS=0



## Citation
```
@inproceedings{wu2024boosting,
  title={Boosting Long-Delayed Reinforcement Learning with Auxiliary Short-Delayed Task},
  author={Wu, Qingyuan and Zhan, Simon Sinong and Wang, Yixuan and Yuhui, Wang and Lin, Chung-Wei and Lv, Chen and Zhu, Qi and Schmidhuber, J{\"u}rgen and Huang, Chao},
  booktitle={International Conference on Machine Learning},
  year={2024},
  organization={PMLR}
}
```


## Acknowledgement

1. CleanRL: https://github.com/vwxyzjn/cleanrl
2. SAC: https://github.com/haarnoja/sac