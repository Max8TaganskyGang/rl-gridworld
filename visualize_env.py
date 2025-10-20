#!/usr/bin/env python3
"""
Visualization script for GridWorld environments (non-interactive version)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from gridworld_env import GridWorldEnv, create_test_environments


def visualize_environment(env, title="GridWorld Environment", save_path=None):
    """Visualize a single GridWorld environment."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create color map
    colors = ['white', 'lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray'] * 4
    colors = colors[:env.n_colors] + ['green', 'black', 'red']  # Add target, wall, obstacle colors
    cmap = ListedColormap(colors)
    
    # Create grid visualization
    grid = np.zeros((env.height, env.width))
    
    for i in range(env.height):
        for j in range(env.width):
            if (i, j) == env.pos_goal:
                grid[i, j] = env.n_colors  # target
            elif env.obstacle_mask[i, j]:
                grid[i, j] = env.n_colors + 2  # obstacle
            else:
                grid[i, j] = env.floor_colors[i, j]
    
    # Display grid
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=env.n_colors + 2)
    
    # Add agent position (if available)
    if env.current_pos is not None:
        agent_row, agent_col = env.current_pos
        ax.add_patch(patches.Circle((agent_col, agent_row), 0.3, color='blue', zorder=10))
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    
    # Add labels
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Column", fontsize=12)
    ax.set_ylabel("Row", fontsize=12)
    
    # Add text annotations
    ax.text(0.02, 0.98, f"Size: {env.height}×{env.width}", transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.text(0.02, 0.90, f"Colors: {env.n_colors}", transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.text(0.02, 0.82, f"Obstacles: {env.obstacle_mask.sum()}", transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add legend
    legend_elements = [
        patches.Patch(color='green', label='Target'),
        patches.Patch(color='black', label='Obstacle'),
        patches.Patch(color='blue', label='Agent'),
        patches.Patch(color='white', label='Floor')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.close(fig)
    return fig, ax


def visualize_all_environments():
    """Visualize all four test environment configurations."""
    print("Creating visualizations for all environments...")
    
    envs = create_test_environments()
    titles = [
        "Environment 1: 5×5, Unique Colors, No Obstacles",
        "Environment 2: 5×5, 5 Colors, No Obstacles", 
        "Environment 3: 10×10, 7 Colors, 10% Obstacles",
        "Environment 4: 10×10, 4 Colors, 10% Obstacles"
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle("GridWorld Environment Configurations", fontsize=16, fontweight='bold')
    
    for i, (env, title) in enumerate(zip(envs, titles)):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Create color map for this environment
        colors = ['white', 'lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray'] * 4
        colors = colors[:env.n_colors] + ['green', 'black', 'red']
        cmap = ListedColormap(colors)
        
        # Create grid visualization
        grid = np.zeros((env.height, env.width))
        
        for h in range(env.height):
            for w in range(env.width):
                if (h, w) == env.pos_goal:
                    grid[h, w] = env.n_colors  # target
                elif env.obstacle_mask[h, w]:
                    grid[h, w] = env.n_colors + 2  # obstacle
                else:
                    grid[h, w] = env.floor_colors[h, w]
        
        # Display grid
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=env.n_colors + 2)
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
        
        # Add labels
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        
        # Add info text
        info_text = f"Size: {env.height}×{env.width}\\nColors: {env.n_colors}\\nObstacles: {env.obstacle_mask.sum()}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add goal marker
        goal_row, goal_col = env.pos_goal
        ax.add_patch(patches.Circle((goal_col, goal_row), 0.2, color='green', zorder=5))
        
        # Add start marker
        start_row, start_col = (0, 0)  # Assuming start is always (0,0)
        ax.add_patch(patches.Circle((start_col, start_row), 0.2, color='blue', zorder=5))
    
    plt.tight_layout()
    plt.savefig('all_environments.png', dpi=150, bbox_inches='tight')
    print("Saved combined visualization to 'all_environments.png'")
    plt.close(fig)


def demonstrate_agent_movement():
    """Demonstrate agent movement in one of the environments."""
    print("Demonstrating agent movement...")
    
    # Create environment 2 (5x5, 5 colors, no obstacles)
    env = GridWorldEnv(height=5, width=5, n_colors=5, seed=42)
    
    obs, info = env.reset()
    print(f"Starting position: {info['current_pos']}")
    
    # Take some steps and visualize each step
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
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Agent Movement Demonstration", fontsize=16, fontweight='bold')
    
    for i, (action, action_name) in enumerate(steps):
        row, col = i // 5, i % 5
        ax = axes[row, col]
        
        # Visualize current state
        colors = ['white', 'lightblue', 'lightgreen', 'lightcoral', 'lightyellow'] + ['green', 'black', 'red']
        cmap = ListedColormap(colors)
        
        grid = np.zeros((env.height, env.width))
        for h in range(env.height):
            for w in range(env.width):
                if (h, w) == env.pos_goal:
                    grid[h, w] = env.n_colors
                else:
                    grid[h, w] = env.floor_colors[h, w]
        
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=env.n_colors + 2)
        
        # Add agent position
        agent_row, agent_col = env.current_pos
        ax.add_patch(patches.Circle((agent_col, agent_row), 0.3, color='blue', zorder=10))
        
        # Add goal
        goal_row, goal_col = env.pos_goal
        ax.add_patch(patches.Circle((goal_col, goal_row), 0.3, color='green', zorder=10))
        
        ax.set_title(f"Step {i+1}: {action_name}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action_name}, Pos={info['current_pos']}, Reward={reward}")
        
        if terminated:
            print("🎉 Goal reached!")
            break
    
    plt.tight_layout()
    plt.savefig('agent_movement.png', dpi=150, bbox_inches='tight')
    print("Saved agent movement visualization to 'agent_movement.png'")
    plt.close(fig)
    
    env.close()


def create_individual_environment_plots():
    """Create individual plots for each environment."""
    print("Creating individual environment plots...")
    
    envs = create_test_environments()
    titles = [
        "Environment 1: 5×5, Unique Colors, No Obstacles",
        "Environment 2: 5×5, 5 Colors, No Obstacles", 
        "Environment 3: 10×10, 7 Colors, 10% Obstacles",
        "Environment 4: 10×10, 4 Colors, 10% Obstacles"
    ]
    
    for i, (env, title) in enumerate(zip(envs, titles)):
        visualize_environment(env, title, f'env_{i+1}.png')
        env.close()


if __name__ == "__main__":
    print("🎨 GridWorld Environment Visualizations")
    print("=" * 50)
    
    # Visualize all environments
    visualize_all_environments()
    
    print("\\n" + "=" * 50)
    
    # Create individual plots
    create_individual_environment_plots()
    
    print("\\n" + "=" * 50)
    
    # Demonstrate agent movement
    demonstrate_agent_movement()
    
    print("\\n🎉 All visualizations completed!")
    print("Generated files:")
    print("- all_environments.png (combined view)")
    print("- env_1.png, env_2.png, env_3.png, env_4.png (individual environments)")
    print("- agent_movement.png (movement demonstration)")