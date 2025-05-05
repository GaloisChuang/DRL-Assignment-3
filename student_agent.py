# student_agent.py
import gym
import cv2
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import Global

# # --- Preprocessing and Frame-Stack ---
# def preprocess_frame(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
#     return resized.astype(np.uint8)

# class FrameStack:
#     def __init__(self, k):
#         self.frames = deque([], maxlen=k)

#     def reset(self, obs):
#         frame = preprocess_frame(obs)
#         self.frames.clear()
#         for _ in range(self.frames.maxlen):
#             self.frames.append(frame)
#         return np.stack(self.frames, axis=0)

#     def step(self, obs):
#         frame = preprocess_frame(obs)
#         self.frames.append(frame)
#         return np.stack(self.frames, axis=0)

# --- Noisy Linear Layer ---
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

# # --- Q-Network ---
class QNet(nn.Module):
    def __init__(self, input_channels, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            NoisyLinear(3136, 512), nn.ReLU()
        )
        self.advantage = NoisyLinear(512, n_actions)
        self.value = NoisyLinear(512, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        adv = self.advantage(x)
        val = self.value(x)
        return val + adv - adv.mean(dim=1, keepdim=True)
    
# def preprocess_batch(batch):
#     """
#     Convert a batch of observations (usually from replay buffer) to float32 and normalize if needed.
#     Ensures normalization only occurs once.
#     """
#     if isinstance(batch, np.ndarray) and batch.dtype == np.uint8:
#         # print("Preprocessing batch to float32 and normalizing.")
#         return torch.tensor(batch, dtype=torch.float32).div(255.0)
#     elif isinstance(batch, torch.Tensor) and batch.dtype == torch.uint8:
#         # print("Preprocessing batch to float32 and normalizing.")
#         return batch.float().div(255.0)
#     elif isinstance(batch, torch.Tensor):
#         return batch  # already float
#     else:
#         raise ValueError("Unexpected input type or dtype in preprocess_batch.")
    
# First = np.load("First_view.npy")

# # --- Agent Wrapper ---
# class Agent:
#     def __init__(self):
#         self.action_space = gym.spaces.Discrete(len(COMPLEX_MOVEMENT))

#     def act(self, obs):
#         # 1) ensure eval mode + disable grads
#         Global.q_net.eval()
#         torch.set_grad_enabled(False)
#         # 2) zero out any NoisyLinear noise
#         for m in Global.q_net.modules():
#             if isinstance(m, Global.NoisyLinear):
#                 m.weight_epsilon.zero_()
#                 m.bias_epsilon.zero_()

#         # 3) frame-skipping + stacking logic
#         if state_is_start_of_episode := np.array_equal(obs, Global.First):
#             Global.state   = Global.stacker.reset(obs)
#             Global.counter = 1
#         elif Global.counter % 4 != 0:
#             Global.counter += 1
#             return Global.action
#         else:
#             Global.state   = Global.stacker.step(obs)
#             Global.counter += 1

#         # 4) forward pass
#         state_tensor = Global.preprocess_batch(Global.state).unsqueeze(0).to(Global.device)
#         with torch.no_grad():
#             Global.action = Global.q_net(state_tensor).argmax(dim=1).item()
#         return Global.action

# ────────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SETUP (do NOT change Agent.__init__)
# ────────────────────────────────────────────────────────────────────────────────

# 1) Load your network & weights once:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_net = QNet(input_channels=4, n_actions=len(COMPLEX_MOVEMENT)).to(device)
ckpt = torch.load("best_agent_9238.pth", map_location=device)
q_net.load_state_dict(ckpt['q_net'])
q_net.eval()

# 2) FrameStacker definition (same as before):
class FrameStacker:
    def __init__(self, k=4):
        self.k = k
        self.frames = deque(maxlen=k)
    def reset(self, obs):
        f = self._proc(obs)
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(f)
        return np.stack(self.frames, axis=0)
    def step(self, obs):
        self.frames.append(self._proc(obs))
        return np.stack(self.frames, axis=0)
    @staticmethod
    def _proc(frame):
        gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84,84), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)

# 3) Keep evaluation state at module scope:
stacker     = FrameStacker(4)
skip        = 4
skip_count  = 0
last_action = 0
state       = None

# ────────────────────────────────────────────────────────────────────────────────
# AGENT CLASS (leave __init__ untouched)
# ────────────────────────────────────────────────────────────────────────────────

class Agent:
    def __init__(self):
        self.action_space = gym.spaces.Discrete(len(COMPLEX_MOVEMENT))

    def act(self, obs: np.ndarray) -> int:
        global skip_count, last_action, state

        # a) first call of episode?
        if state is None:
            state      = stacker.reset(obs)
            skip_count = 1
        else:
            # b) frame-skip
            if skip_count < skip:
                skip_count += 1
                return last_action
            skip_count = 1
            state = stacker.step(obs)

        # c) forward pass (eval mode, no noise reset needed)
        tensor = (torch.tensor(state, dtype=torch.float32)
                        .div(255.0)
                        .unsqueeze(0)
                        .to(device))
        with torch.no_grad():
            last_action = q_net(tensor).argmax(dim=1).item()

        return last_action