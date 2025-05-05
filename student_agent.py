# student_agent/agent.py
import numpy as np
import torch
import gym
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import Global

class Agent:
    def __init__(self):
        self.action_space = gym.spaces.Discrete(len(COMPLEX_MOVEMENT))

    def act(self, obs):
        # 1) ensure eval mode + disable grads
        Global.q_net.eval()
        torch.set_grad_enabled(False)
        # 2) zero out any NoisyLinear noise
        for m in Global.q_net.modules():
            if isinstance(m, Global.NoisyLinear):
                m.weight_epsilon.zero_()
                m.bias_epsilon.zero_()

        # 3) frame-skipping + stacking logic
        if state_is_start_of_episode := np.array_equal(obs, Global.First):
            Global.state   = Global.stacker.reset(obs)
            Global.counter = 1
        elif Global.counter % 4 != 0:
            Global.counter += 1
            return Global.action
        else:
            Global.state   = Global.stacker.step(obs)
            Global.counter += 1

        # 4) forward pass
        state_tensor = Global.preprocess_batch(Global.state).unsqueeze(0).to(Global.device)
        with torch.no_grad():
            Global.action = Global.q_net(state_tensor).argmax(dim=1).item()
        return Global.action
