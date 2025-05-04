import cv2
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

# --- Preprocessing and Frame-Stack ---
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)

class FrameStack:
    def __init__(self, k):
        self.frames = deque([], maxlen=k)

    def reset(self, obs):
        frame = preprocess_frame(obs)
        self.frames.clear()
        for _ in range(self.frames.maxlen):
            self.frames.append(frame)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = preprocess_frame(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

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

# --- Q-Network ---
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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

q_net = QNet(input_channels=4, n_actions=len(COMPLEX_MOVEMENT)).to(device)
ckpt = torch.load("best_agent.pth", map_location=device)
q_net.load_state_dict(ckpt['q_net'])
q_net.eval()

stacker = FrameStack(k=4)

state   = None
counter = 0
action  = None