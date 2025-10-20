import argparse
import numpy as np

from gridworld_env import GridWorldEnv, create_test_environments

# Algorithms
from algorithms.dqn import DQNAgent, train_dqn
from algorithms.ppo import PPOAgent, PPOConfig, train_ppo


def make_env(env_id: int) -> GridWorldEnv:
	envs = create_test_environments()
	if env_id < 1 or env_id > len(envs):
		raise ValueError(f"env_id must be in [1,{len(envs)}]")
	return envs[env_id - 1]


def run_dqn(args):
	env = make_env(args.env_id)
	state_size = env.observation_space.n
	action_size = env.action_space.n
	agent = DQNAgent(
		state_size=state_size,
		action_size=action_size,
		learning_rate=args.lr,
		buffer_size=args.buffer_size,
		batch_size=args.batch_size,
	)
	rewards, losses = train_dqn(env, agent, episodes=args.episodes, max_steps=args.max_steps, 
	                           use_wandb=args.wandb, project_name=f"rl-gridworld-dqn-env{args.env_id}")
	print(f"DQN finished. Avg reward (last 50): {np.mean(rewards[-50:]):.3f}")
	env.close()


def run_ppo(args):
	env = make_env(args.env_id)
	state_size = env.observation_space.n
	action_size = env.action_space.n
	cfg = PPOConfig(learning_rate=args.lr, minibatch_size=args.batch_size)
	agent = PPOAgent(state_size=state_size, action_size=action_size, config=cfg)
	rewards = train_ppo(env, agent, total_steps=args.total_steps, rollout_len=args.rollout_len,
	                   use_wandb=args.wandb, project_name=f"rl-gridworld-ppo-env{args.env_id}")
	print(f"PPO finished. Avg rollout reward (last 10): {np.mean(rewards[-10:]):.3f}")
	env.close()


def parse_args():
	p = argparse.ArgumentParser(description="Train RL agents on GridWorld")
	p.add_argument("algo", choices=["dqn", "ppo"], help="Algorithm to train")
	p.add_argument("--env-id", type=int, default=2, help="Environment id (1..4)")
	# common
	p.add_argument("--lr", type=float, default=3e-4)
	p.add_argument("--batch-size", type=int, default=64)
	p.add_argument("--wandb", action="store_true", help="Enable wandb logging")
	# dqn
	p.add_argument("--episodes", type=int, default=500)
	p.add_argument("--max-steps", type=int, default=100)
	p.add_argument("--buffer-size", type=int, default=50000)
	# ppo
	p.add_argument("--total-steps", type=int, default=10000)
	p.add_argument("--rollout-len", type=int, default=256)
	return p.parse_args()


def main():
	args = parse_args()
	if args.algo == "dqn":
		run_dqn(args)
	elif args.algo == "ppo":
		run_ppo(args)


if __name__ == "__main__":
	main()
