import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import Tuple, List
import wandb


class DQN(nn.Module):
	"""Deep Q-Network for discrete observation/action spaces."""

	def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = 4):
		super().__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, output_size)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return self.fc3(x)


class ReplayBuffer:
	"""Experience replay buffer for DQN."""

	def __init__(self, capacity: int = 10000):
		self.buffer = deque(maxlen=capacity)

	def push(self, state, action, reward, next_state, done):
		self.buffer.append((state, action, reward, next_state, done))

	def sample(self, batch_size: int) -> Tuple:
		batch = random.sample(self.buffer, batch_size)
		state, action, reward, next_state, done = map(np.stack, zip(*batch))
		return state, action, reward, next_state, done

	def __len__(self):
		return len(self.buffer)


class DQNAgent:
	"""DQN Agent wrapper with training loop utilities."""

	def __init__(
		self,
		state_size: int,
		action_size: int,
		learning_rate: float = 1e-3,
		gamma: float = 0.99,
		epsilon_start: float = 1.0,
		epsilon_end: float = 0.05,
		epsilon_decay: float = 0.995,
		buffer_size: int = 50000,
		batch_size: int = 64,
		target_update_interval: int = 200,
		device: str | None = None,
	):
		self.state_size = state_size
		self.action_size = action_size
		self.gamma = gamma
		self.epsilon = epsilon_start
		self.epsilon_end = epsilon_end
		self.epsilon_decay = epsilon_decay
		self.batch_size = batch_size
		self.target_update_interval = target_update_interval

		self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.q_net = DQN(state_size, output_size=action_size).to(self.device)
		self.target_net = DQN(state_size, output_size=action_size).to(self.device)
		self.target_net.load_state_dict(self.q_net.state_dict())
		self.target_net.eval()

		self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
		self.memory = ReplayBuffer(buffer_size)

		self.train_steps = 0

	def one_hot(self, s: int) -> np.ndarray:
		v = np.zeros(self.state_size, dtype=np.float32)
		v[s] = 1.0
		return v

	@torch.no_grad()
	def act(self, state: int, explore: bool = True) -> int:
		if explore and np.random.rand() < self.epsilon:
			return np.random.randint(self.action_size)
		state_t = torch.from_numpy(self.one_hot(state)).unsqueeze(0).to(self.device)
		q = self.q_net(state_t)
		return int(q.argmax(dim=1).item())

	def remember(self, s, a, r, s2, done):
		self.memory.push(s, a, r, s2, done)

	def train_step(self) -> float:
		if len(self.memory) < self.batch_size:
			return 0.0
		states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
		states_t = torch.from_numpy(np.stack([self.one_hot(s) for s in states]).astype(np.float32)).to(self.device)
		actions_t = torch.from_numpy(actions.astype(np.int64)).to(self.device).unsqueeze(1)
		rewards_t = torch.from_numpy(rewards.astype(np.float32)).to(self.device)
		next_states_t = torch.from_numpy(np.stack([self.one_hot(s) for s in next_states]).astype(np.float32)).to(self.device)
		dones_t = torch.from_numpy(dones.astype(np.bool_)).to(self.device)

		q_values = self.q_net(states_t).gather(1, actions_t).squeeze(1)
		with torch.no_grad():
			next_q = self.target_net(next_states_t).max(dim=1)[0]
			target = rewards_t + self.gamma * next_q * (~dones_t)

		loss = F.mse_loss(q_values, target)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.train_steps += 1
		if self.train_steps % self.target_update_interval == 0:
			self.target_net.load_state_dict(self.q_net.state_dict())

		# epsilon decay
		self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
		return float(loss.item())


def train_dqn(env, agent: DQNAgent, episodes: int = 500, max_steps: int = 100, 
              use_wandb: bool = True, project_name: str = "rl-gridworld-dqn") -> tuple[list[float], list[float]]:
	"""Training loop with wandb logging."""
	
	if use_wandb:
		wandb.init(
			project=project_name,
			config={
				"algorithm": "DQN",
				"episodes": episodes,
				"max_steps": max_steps,
				"learning_rate": agent.optimizer.param_groups[0]['lr'],
				"gamma": agent.gamma,
				"epsilon_start": agent.epsilon,
				"batch_size": agent.batch_size,
				"buffer_size": len(agent.memory.buffer),
				"env_size": f"{env.height}x{env.width}",
				"n_colors": env.n_colors,
				"obstacles": env.obstacle_mask.sum()
			}
		)
	
	rewards: list[float] = []
	losses: list[float] = []
	successes: list[bool] = []
	
	for episode in range(episodes):
		state, _ = env.reset()
		episode_reward = 0.0
		acc_loss = 0.0
		steps = 0
		
		for step in range(max_steps):
			action = agent.act(state, explore=True)
			next_state, reward, terminated, truncated, _ = env.step(action)
			done = terminated or truncated
			agent.remember(state, action, reward, next_state, done)
			loss = agent.train_step()
			acc_loss += loss
			episode_reward += reward
			state = next_state
			steps += 1
			if done:
				break
		
		rewards.append(episode_reward)
		losses.append(acc_loss / max(steps, 1))
		successes.append(episode_reward > 0)
		
		# Log to wandb every 10 episodes
		if use_wandb and episode % 10 == 0:
			avg_reward = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
			avg_loss = np.mean(losses[-10:]) if len(losses) >= 10 else np.mean(losses)
			success_rate = np.mean(successes[-10:]) if len(successes) >= 10 else np.mean(successes)
			
			wandb.log({
				"episode": episode,
				"episode_reward": episode_reward,
				"avg_reward_10": avg_reward,
				"episode_loss": acc_loss / max(steps, 1),
				"avg_loss_10": avg_loss,
				"success_rate_10": success_rate,
				"epsilon": agent.epsilon,
				"steps": steps,
				"buffer_size": len(agent.memory)
			})
	
	if use_wandb:
		wandb.finish()
	
	return rewards, losses
