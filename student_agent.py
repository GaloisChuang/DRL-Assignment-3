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
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        # if self.training:
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        # else:
        #     weight = self.weight_mu
        #     bias = self.bias_mu
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

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

# --- Prioritized Replay Buffer ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, n_step=3, gamma=0.99):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)

    def add(self, transition):
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_step:
            return
        # compute n-step return
        reward, next_state, done = self._get_n_step_info()
        state, action, _, _, _ = self.n_step_buffer[0]
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max(self.priorities, default=1.0))

    def _get_n_step_info(self):
        reward, next_state, done = self.n_step_buffer[-1][2:5]
        for (s, a, r, ns, d) in reversed(list(self.n_step_buffer)[:-1]):
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (ns, d) if d else (next_state, done)
        return reward, next_state, done

    def sample(self, batch_size, beta=0.4):
        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        states, actions, rewards, next_states, dones = zip(*samples)
        states = np.stack(states, axis=0)
        next_states = np.stack(next_states, axis=0)
        return states, actions, rewards, next_states, dones, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, priorities):
        for idx, p in zip(indices, priorities):
            self.priorities[idx] = float(p)

    def __len__(self):
        return len(self.buffer)

# --- Batch Preprocessing ---
def preprocess_batch(batch):
    if isinstance(batch, np.ndarray) and batch.dtype == np.uint8:
        return torch.tensor(batch, dtype=torch.float32).div(255.0)
    if isinstance(batch, torch.Tensor):
        return batch.float().div(255.0) if batch.dtype == torch.uint8 else batch
    raise ValueError(f"Unexpected batch type {type(batch)}")

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, state_shape, n_actions, gamma=0.99, lr=2.5e-4, batch_size=64,
                buffer_size=10000, tau=0.01, device='cuda'):
        self.device = torch.device(device)
        self.q_net = QNet(state_shape[0], n_actions).to(self.device)
        self.target_net = QNet(state_shape[0], n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.RMSprop(
            self.q_net.parameters(), lr=lr, alpha=0.95, eps=1e-2)
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, n_step=5, gamma=gamma)
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

    def get_action(self, state, eval_mode=False):
        # state: np.ndarray (4,84,84), uint8
        s = preprocess_batch(state).unsqueeze(0).to(self.device)
        self.q_net.eval() if eval_mode else self.q_net.train()
        with torch.no_grad():
            return self.q_net(s).argmax(1).item()

    def update(self):
        for t, s in zip(self.target_net.parameters(), self.q_net.parameters()):
            t.data.copy_(self.tau * s.data + (1-self.tau) * t.data)

    def train(self, beta=0.4):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones, idxs, weights = \
            self.replay_buffer.sample(self.batch_size, beta)
        s = preprocess_batch(states).to(self.device)
        ns = preprocess_batch(next_states).to(self.device)
        a = torch.tensor(actions, dtype=torch.int64).to(self.device)
        r = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        d = torch.tensor(dones, dtype=torch.float32).to(self.device)
        w = weights.to(self.device)
        # Q values
        q_vals = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        # next
        with torch.no_grad():
            next_q = self.q_net(ns).argmax(dim=1, keepdim=True)
            target_q = self.target_net(ns).gather(1, next_q).squeeze(1)
            q_target = r + (1-d)*(self.gamma**self.replay_buffer.n_step)*target_q
        td = q_vals - q_target
        loss = (td.pow(2) * w).mean()
        self.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self.replay_buffer.update_priorities(idxs, td.abs().detach().cpu().numpy())

# --- Agent Wrapper ---
class Agent:
    def __init__(self):
        self.action_space = gym.spaces.Discrete(len(COMPLEX_MOVEMENT))

    # def act(self, obs):
    #     if Global.counter % 4 != 0 and Global.action is not None:
    #         Global.counter += 1
    #         return Global.action

    #     # 2) build 4-frame state
    #     if Global.state is None or np.array_equal(obs, Global.first):
    #         Global.state = Global.stacker.reset(obs)
    #         # Global.counter = 0
    #         print("New episode started")
    #         print(Global.counter)
    #     else:
    #         Global.state = Global.stacker.step(obs)

    #     # 3) preprocess and forward through Q-net
    #     state_tensor = torch.tensor(Global.state, dtype=torch.float32).div(255.0)
    #     state_tensor = state_tensor.unsqueeze(0).to(Global.device)  # [1,4,84,84]
    #     Global.q_net.eval()     
    #     with torch.no_grad():
    #         q_vals = Global.q_net(state_tensor)                    # [1, n_actions]
    #     best_a = q_vals.argmax(dim=1).item()

    #     # 4) store & return
    #     Global.action = best_a
    #     Global.counter += 1
    #     return best_a
    

    def act(self, obs):
        if np.array_equal(obs, Global.first):
            if Global.state is None or not np.array_equal(preprocess_frame(obs), Global.state[-1]):
                print("New episode started")
                Global.state = Global.stacker.reset(obs)
                Global.counter = 0
                state_tensor = torch.tensor(Global.state, dtype=torch.float32).div(255.0)
                state_tensor = state_tensor.unsqueeze(0).to(Global.device)  # [1,4,84,84]
                Global.q_net.eval()     
                with torch.no_grad():
                    q_vals = Global.q_net(state_tensor)                    # [1, n_actions]
                action = q_vals.argmax(dim=1).item()
                Global.action = action
                Global.counter += 1
                # print("Case 1")
                return action
        if Global.counter % 4 != 0 and Global.action is not None:
            Global.counter += 1
            # print("Case 2")
            return Global.action
        else:
            Global.state = Global.stacker.step(obs)
            state_tensor = torch.tensor(Global.state, dtype=torch.float32).div(255.0)
            state_tensor = state_tensor.unsqueeze(0).to(Global.device)
            Global.q_net.eval()
            with torch.no_grad():
                q_vals = Global.q_net(state_tensor)
            action = q_vals.argmax(dim=1).item()
            Global.action = action
            Global.counter += 1
            # print("Case 3")
            return action