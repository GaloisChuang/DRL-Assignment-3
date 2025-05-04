import cv2
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from Global import *

# --- Preprocessing and Lazy Frame-Stack for Memory Efficiency ---
class LazyFrames:
    """Efficient storage of frame-stacks to avoid duplicate copies."""
    def __init__(self, frames):
        self.frames = frames

    def __array__(self, dtype=None):
        arr = np.stack(self.frames, axis=0)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)


class FrameStack:
    def __init__(self, k):
        self.frames = deque([], maxlen=k)

    def reset(self, obs):
        processed = preprocess_frame(obs)
        for _ in range(self.frames.maxlen):
            self.frames.append(processed)
        return LazyFrames(list(self.frames))

    def step(self, obs):
        processed = preprocess_frame(obs)
        self.frames.append(processed)
        return LazyFrames(list(self.frames))

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

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = torch.randn(self.in_features)
        epsilon_out = torch.randn(self.out_features)
        # print(f"epsilon_in: {epsilon_in.shape}, epsilon_out: {epsilon_out.shape}")
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)
    

# --- Q-Network Definition ---
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
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

# --- Replay Buffer with LazyFrames ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, n_step=3, gamma=0.99):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.pos = 0
        self.alpha = alpha
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)

    def add(self, transition):
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_step:
            return

        reward, next_state, done = self._get_n_step_info()
        state, action, _, _, _ = self.n_step_buffer[0]
        n_step_transition = (state, action, reward, next_state, done)

        max_prio = max(self.priorities, default=1.0)
        self.buffer.append(n_step_transition)
        self.priorities.append(max_prio)

    def _get_n_step_info(self):
        reward, next_state, done = self.n_step_buffer[-1][2], self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[2], transition[3], transition[4]
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)
        return reward, next_state, done

    def sample(self, batch_size, beta=0.4):
        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        states = np.stack([np.array(s, dtype=np.uint8) for s in states])
        next_states = np.stack([np.array(s, dtype=np.uint8) for s in next_states])

        return states, actions, rewards, next_states, dones, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = float(prio)

    def __len__(self):
        return len(self.buffer)
    
def preprocess_batch(batch):
    """
    Convert a batch of observations (usually from replay buffer) to float32 and normalize if needed.
    Ensures normalization only occurs once.
    """
    if isinstance(batch, np.ndarray) and batch.dtype == np.uint8:
        # print("Preprocessing batch to float32 and normalizing.")
        return torch.tensor(batch, dtype=torch.float32).div(255.0)
    elif isinstance(batch, torch.Tensor) and batch.dtype == torch.uint8:
        # print("Preprocessing batch to float32 and normalizing.")
        return batch.float().div(255.0)
    elif isinstance(batch, torch.Tensor):
        return batch  # already float
    else:
        raise ValueError("Unexpected input type or dtype in preprocess_batch.")

# --- DQN Agent incorporating ICM ---
class DQNAgent:
    def __init__(self, state_shape, n_actions,
                 gamma=0.99, learning_rate=2.5e-4, batch_size=64, buffer_size=10000, tau=0.01,
                 device='cuda', q_net=None, target_net=None):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tau = tau
        self.device = torch.device(device)

        self.q_net = QNet(state_shape[0], n_actions).to(self.device) if q_net is None else q_net
        self.target_net = QNet(state_shape[0], n_actions).to(self.device) if target_net is None else target_net
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.RMSprop(
            list(self.q_net.parameters()),
            lr=2.5e-4, alpha=0.95, eps=1e-2
        )
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=0.6, n_step=5, gamma=self.gamma)

    def get_action(self, state, eval_mode=False):
        if isinstance(state, LazyFrames):
            state = np.array(state, dtype=np.uint8)
        state_tensor = preprocess_batch(state).unsqueeze(0).to(self.device)
        assert state_tensor.max() <= 1.0 and state_tensor.min() >= 0.0, "State tensor should be normalized between 0 and 1."
        self.q_net.eval() if eval_mode else self.q_net.train()
        with torch.no_grad():
            return self.q_net(state_tensor).argmax(1).item()

    def update(self):
        for t, s in zip(self.target_net.parameters(), self.q_net.parameters()):
            t.data.copy_(self.tau * s.data + (1.0 - self.tau) * t.data)

    def train(self, beta=0.4):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size, beta)
        device = self.device

        s = preprocess_batch(states).to(device)
        assert s.max() <= 1.0 and s.min() >= 0.0, "State tensor should be normalized between 0 and 1."
        ns = preprocess_batch(next_states).to(device)
        assert ns.max() <= 1.0 and ns.min() >= 0.0, "Next state tensor should be normalized between 0 and 1."
        a = torch.tensor(actions, dtype=torch.int64).to(device)
        r = torch.tensor(rewards, dtype=torch.float32).to(device)
        d = torch.tensor(dones, dtype=torch.float32).to(device)
        weights = weights.to(device)

        # Current Q values
        q_vals = self.q_net(s)
        q_val = q_vals.gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN: argmax from q_net, value from target_net
        with torch.no_grad():
            next_q_vals = self.q_net(ns)
            next_actions = next_q_vals.argmax(dim=1, keepdim=True)
            next_target_q = self.target_net(ns).gather(1, next_actions).squeeze(1)
            q_target = r + (1 - d) * (self.gamma ** self.replay_buffer.n_step) * next_target_q

        td_error = q_val - q_target
        loss = (td_error ** 2 * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        # self.update()

        # Update PER priorities
        new_priorities = td_error.abs().detach().cpu() + 1e-6
        self.replay_buffer.update_priorities(indices, new_priorities)

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)

    def act(self, observation):
        Global.counter += 1
        if Global.counter % 4 != 0 and Global.action is not None:
            return Global.action
        if Global.state is None:
            Global.state = stacker.reset(observation)
        else:
            Global.state = stacker.step(observation)
        agent = Global.agent
        action = agent.get_action(Global.state, eval_mode=True)
        Global.action = action
        
        