#!/usr/bin/env python3
"""
Test script for GridWorld environment
"""

from gridworld_env import GridWorldEnv, create_test_environments
import numpy as np

def test_environment():
    """Test basic environment functionality."""
    print("Testing GridWorld Environment...")
    
    # Create a simple environment
    env = GridWorldEnv(height=5, width=5, n_colors=5, seed=42)
    
    # Test reset
    obs, info = env.reset()
    print(f"✓ Reset successful: obs={obs}, pos={info['current_pos']}")
    
    # Test step
    action = 1  # Move right
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Step successful: action={action}, obs={obs}, reward={reward}")
    
    # Test multiple steps
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    
    print(f"✓ Multiple steps: total_reward={total_reward}, final_pos={info['current_pos']}")
    
    env.close()
    print("✓ Environment closed successfully")

def test_all_environments():
    """Test all four environment configurations."""
    print("\nTesting all environment configurations...")
    
    envs = create_test_environments()
    
    for i, env in enumerate(envs):
        print(f"\nEnvironment {i+1}:")
        print(f"  Size: {env.height}x{env.width}")
        print(f"  Colors: {env.n_colors}")
        print(f"  Obstacles: {env.obstacle_mask.sum()}")
        
        # Quick test
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  ✓ Test passed: obs={obs}, reward={reward}")
        
        env.close()

if __name__ == "__main__":
    test_environment()
    test_all_environments()
    print("\n🎉 All tests passed!")
