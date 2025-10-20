#!/usr/bin/env python3
"""
ASCII visualization of GridWorld environments for terminal display
"""

import numpy as np
from gridworld_env import GridWorldEnv, create_test_environments


def print_environment_ascii(env, title="GridWorld Environment"):
    """Print environment as ASCII art in terminal."""
    print(f"\n{title}")
    print("=" * len(title))
    
    # Create ASCII representation
    grid = []
    for i in range(env.height):
        row = []
        for j in range(env.width):
            if (i, j) == env.pos_goal:
                row.append('G')  # Goal
            elif env.obstacle_mask[i, j]:
                row.append('█')  # Obstacle
            else:
                # Use different symbols for different floor colors
                color_symbols = ['·', '○', '●', '◐', '◑', '◒', '◓', '◔', '◕', '◖', '◗', '◘', '◙', '◚', '◛', '◜', '◝', '◞', '◟', '◠', '◡', '◢', '◣', '◤', '◥']
                color_idx = env.floor_colors[i, j] % len(color_symbols)
                row.append(color_symbols[color_idx])
        grid.append(row)
    
    # Print grid with borders
    print("┌" + "─" * (env.width * 2 + 1) + "┐")
    for i, row in enumerate(grid):
        print("│ " + " ".join(row) + " │")
    print("└" + "─" * (env.width * 2 + 1) + "┘")
    
    # Print legend
    print(f"Size: {env.height}×{env.width}")
    print(f"Colors: {env.n_colors}")
    print(f"Obstacles: {env.obstacle_mask.sum()}")
    print("Legend: G=Goal, █=Obstacle, ·○●=Floor colors")


def print_agent_movement():
    """Demonstrate agent movement with ASCII visualization."""
    print("\n🤖 Agent Movement Demonstration")
    print("=" * 40)
    
    env = GridWorldEnv(height=5, width=5, n_colors=5, seed=42)
    obs, info = env.reset()
    
    print(f"Starting position: {info['current_pos']}")
    
    # Movement sequence
    steps = [
        (1, "Right"),
        (1, "Right"), 
        (2, "Down"),
        (2, "Down"),
        (1, "Right"),
        (1, "Right"),
        (0, "Up"),
        (0, "Up"),
        (0, "Up"),
        (0, "Up")
    ]
    
    for i, (action, action_name) in enumerate(steps):
        print(f"\nStep {i+1}: {action_name}")
        
        # Create ASCII grid with agent position
        grid = []
        for h in range(env.height):
            row = []
            for w in range(env.width):
                if (h, w) == env.pos_goal:
                    row.append('G')
                elif env.obstacle_mask[h, w]:
                    row.append('█')
                elif (h, w) == env.current_pos:
                    row.append('A')  # Agent
                else:
                    color_symbols = ['·', '○', '●', '◐', '◑']
                    color_idx = env.floor_colors[h, w] % len(color_symbols)
                    row.append(color_symbols[color_idx])
            grid.append(row)
        
        # Print grid
        print("┌" + "─" * (env.width * 2 + 1) + "┐")
        for row in grid:
            print("│ " + " ".join(row) + " │")
        print("└" + "─" * (env.width * 2 + 1) + "┘")
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Position: {info['current_pos']}, Reward: {reward}")
        
        if terminated:
            print("🎉 Goal reached!")
            break
    
    env.close()


def print_all_environments():
    """Print all four environments as ASCII art."""
    print("\n🌍 All GridWorld Environment Configurations")
    print("=" * 50)
    
    envs = create_test_environments()
    titles = [
        "Environment 1: 5×5, Unique Colors, No Obstacles",
        "Environment 2: 5×5, 5 Colors, No Obstacles", 
        "Environment 3: 10×10, 7 Colors, 10% Obstacles",
        "Environment 4: 10×10, 4 Colors, 10% Obstacles"
    ]
    
    for i, (env, title) in enumerate(zip(envs, titles)):
        print_environment_ascii(env, title)
        env.close()
        if i < len(envs) - 1:
            print("\n" + "-" * 50)


if __name__ == "__main__":
    print("🎨 GridWorld ASCII Visualizations")
    print("=" * 50)
    
    # Print all environments
    print_all_environments()
    
    # Demonstrate agent movement
    print_agent_movement()
    
    print("\n🎉 ASCII visualizations completed!")
    print("\nNote: PNG files are also available:")
    print("- all_environments.png")
    print("- env_1.png, env_2.png, env_3.png, env_4.png") 
    print("- agent_movement.png")
