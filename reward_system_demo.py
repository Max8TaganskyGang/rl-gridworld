#!/usr/bin/env python3
"""
Demonstration of different reward systems in GridWorld
"""

import numpy as np
from gridworld_env import GridWorldEnv


def test_reward_system(env, name, steps=20):
    """Test a reward system and print statistics."""
    print(f"\n=== {name} ===")
    
    rewards = []
    positions = []
    
    state, _ = env.reset()
    positions.append(env.current_pos)
    
    for i in range(steps):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        
        rewards.append(reward)
        positions.append(info['current_pos'])
        
        if terminated:
            print(f"🎉 Goal reached at step {i+1}!")
            break
        elif truncated:
            print(f"⏰ Max steps reached at step {i+1}")
            break
    
    # Calculate statistics
    total_reward = sum(rewards)
    avg_reward = np.mean(rewards)
    max_reward = max(rewards) if rewards else 0
    min_reward = min(rewards) if rewards else 0
    
    print(f"Total reward: {total_reward:.3f}")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Max reward: {max_reward:.3f}")
    print(f"Min reward: {min_reward:.3f}")
    print(f"Steps taken: {len(rewards)}")
    
    return rewards, positions


def main():
    print("🎯 GridWorld Reward System Comparison")
    print("=" * 50)
    
    # Test different reward configurations
    configs = [
        {
            "name": "Sparse Rewards (Original)",
            "goal_reward": 1.0,
            "step_penalty": 0.0,
            "collision_penalty": 0.0,
            "distance_reward_scale": 0.0
        },
        {
            "name": "Dense Rewards (New)",
            "goal_reward": 10.0,
            "step_penalty": -0.01,
            "collision_penalty": -0.1,
            "distance_reward_scale": 0.1
        },
        {
            "name": "Efficiency Focused",
            "goal_reward": 5.0,
            "step_penalty": -0.05,
            "collision_penalty": -0.2,
            "distance_reward_scale": 0.05
        },
        {
            "name": "Exploration Focused",
            "goal_reward": 20.0,
            "step_penalty": -0.001,
            "collision_penalty": -0.05,
            "distance_reward_scale": 0.2
        }
    ]
    
    for config in configs:
        env = GridWorldEnv(
            height=5, width=5, n_colors=5, seed=42,
            goal_reward=config["goal_reward"],
            step_penalty=config["step_penalty"],
            collision_penalty=config["collision_penalty"],
            distance_reward_scale=config["distance_reward_scale"]
        )
        
        rewards, positions = test_reward_system(env, config["name"])
        env.close()
    
    print("\n" + "=" * 50)
    print("📊 Reward System Analysis:")
    print("• Sparse: Only rewards reaching goal (hard to learn)")
    print("• Dense: Rewards progress + penalizes inefficiency (balanced)")
    print("• Efficiency: Strong penalties for wasted steps (fast learning)")
    print("• Exploration: High rewards for getting closer (encourages exploration)")


if __name__ == "__main__":
    main()
