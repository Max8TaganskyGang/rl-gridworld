import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import Tuple


class ActorCritic(nn.Module):
	"""Shared MLP with separate policy and value heads for discrete action spaces."""

	def __init__(self, input_size: int, hidden_size: int = 128, action_size: int = 4):
		super().__init__()
		self.body = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.Tanh(),
			nn.Linear(hidden_size, hidden_size),
			nn.Tanh(),
		)
		self.policy = nn.Linear(hidden_size, action_size)
		self.value = nn.Linear(hidden_size, 1)

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		h = self.body(x)
		logits = self.policy(h)
		value = self.value(h).squeeze(-1)
		return logits, value


@dataclass
class PPOConfig:
	learning_rate: float = 3e-4
	gamma: float = 0.99
	gae_lambda: float = 0.95
	clip_range: float = 0.2
	entropy_coef: float = 0.01
	value_coef: float = 0.5
	max_grad_norm: float = 0.5
	update_epochs: int = 4
	minibatch_size: int = 64


class PPOAgent:
	def __init__(self, state_size: int, action_size: int, config: PPOConfig | None = None, device: str | None = None):
		self.state_size = state_size
		self.action_size = action_size
		self.cfg = config or PPOConfig()
		self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.net = ActorCritic(state_size, action_size=action_size).to(self.device)
		self.optimizer = optim.Adam(self.net.parameters(), lr=self.cfg.learning_rate)

	def one_hot(self, s: int) -> np.ndarray:
		v = np.zeros(self.state_size, dtype=np.float32)
		v[s] = 1.0
		return v

	@torch.no_grad()
	def act(self, state: int):
		s = torch.from_numpy(self.one_hot(state)).unsqueeze(0).to(self.device)
		logits, value = self.net(s)
		probs = torch.distributions.Categorical(logits=logits)
		action = probs.sample()
		log_prob = probs.log_prob(action)
		return int(action.item()), float(log_prob.item()), float(value.item())

	def update(self, batch):
		states, actions, old_log_probs, returns, advantages, values = batch
		states = torch.from_numpy(states).to(self.device)
		actions = torch.from_numpy(actions).to(self.device)
		old_log_probs = torch.from_numpy(old_log_probs).to(self.device)
		returns = torch.from_numpy(returns).to(self.device)
		advantages = torch.from_numpy(advantages).to(self.device)
		advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

		loss_pi_total = 0.0
		loss_v_total = 0.0
		entropy_total = 0.0

		for _ in range(self.cfg.update_epochs):
			idx = np.arange(states.size(0))
			np.random.shuffle(idx)
			for start in range(0, len(idx), self.cfg.minibatch_size):
				mb = idx[start : start + self.cfg.minibatch_size]
				mb_states = states[mb]
				mb_actions = actions[mb]
				mb_old_logp = old_log_probs[mb]
				mb_returns = returns[mb]
				mb_adv = advantages[mb]

				logits, values = self.net(mb_states)
				dist = torch.distributions.Categorical(logits=logits)
				logp = dist.log_prob(mb_actions)
				entropy = dist.entropy().mean()

				ratio = torch.exp(logp - mb_old_logp)
				clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range) * mb_adv
				loss_pi = -(torch.min(ratio * mb_adv, clipped)).mean()
				loss_v = torch.mean((mb_returns - values) ** 2)
				loss = loss_pi + self.cfg.value_coef * loss_v - self.cfg.entropy_coef * entropy

				self.optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
				self.optimizer.step()

				loss_pi_total += float(loss_pi.item())
				loss_v_total += float(loss_v.item())
				entropy_total += float(entropy.item())

		return {
			"policy_loss": loss_pi_total,
			"value_loss": loss_v_total,
			"entropy": entropy_total,
		}


def rollout_trajectory(env, agent: PPOAgent, steps: int):
	obs_list = []
	actions = []
	log_probs = []
	rewards = []
	values = []
	dones = []

	state, _ = env.reset()
	for _ in range(steps):
		action, logp, value = agent.act(state)
		next_state, reward, terminated, truncated, _ = env.step(action)
		done = terminated or truncated

		obs_list.append(agent.one_hot(state))
		actions.append(action)
		log_probs.append(logp)
		rewards.append(reward)
		values.append(value)
		dones.append(done)

		state = next_state if not done else env.reset()[0]

	return (
		np.array(obs_list, dtype=np.float32),
		np.array(actions, dtype=np.int64),
		np.array(log_probs, dtype=np.float32),
		np.array(rewards, dtype=np.float32),
		np.array(values, dtype=np.float32),
		np.array(dones, dtype=np.bool_),
	)


def compute_gae(rewards, values, dones, gamma: float, lam: float):
	T = len(rewards)
	deltas = np.zeros(T, dtype=np.float32)
	advantages = np.zeros(T, dtype=np.float32)
	next_value = 0.0
	adv = 0.0
	for t in reversed(range(T)):
		delta = rewards[t] + gamma * next_value * (1.0 - float(dones[t])) - values[t]
		adv = delta + gamma * lam * (1.0 - float(dones[t])) * adv
		advantages[t] = adv
		next_value = values[t]
	returns = advantages + values
	return advantages, returns


def train_ppo(env, agent: PPOAgent, total_steps: int = 10_000, rollout_len: int = 256):
	rewards_per_rollout: list[float] = []
	steps = 0
	state, _ = env.reset()
	while steps < total_steps:
		obs, actions, logp, rewards, values, dones = rollout_trajectory(env, agent, steps=min(rollout_len, total_steps - steps))
		advantages, returns = compute_gae(rewards, values, dones, agent.cfg.gamma, agent.cfg.gae_lambda)
		stats = agent.update((obs, actions, logp, returns, advantages, values))
		rewards_per_rollout.append(float(np.sum(rewards)))
		steps += len(rewards)
	return rewards_per_rollout
