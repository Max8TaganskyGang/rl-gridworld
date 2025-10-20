#!/usr/bin/env python3
"""
Improved visualization script for GridWorld environments
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from gridworld_env import GridWorldEnv, create_test_environments, create_random_obstacle_mask


def visualize_environment_improved(env, title="GridWorld Environment", save_path=None):
    """Visualize a single GridWorld environment with improved markers."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create color map with more distinct colors
    colors = ['white', 'lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 
              'lightpink', 'lightgray', 'wheat', 'lavender', 'mistyrose'] * 3
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
    
    # Add agent position (always visible with better styling)
    if env.current_pos is not None:
        agent_row, agent_col = env.current_pos
        # Large blue circle with white border
        ax.add_patch(patches.Circle((agent_col, agent_row), 0.4, color='white', zorder=15))
        ax.add_patch(patches.Circle((agent_col, agent_row), 0.35, color='blue', zorder=16))
        ax.add_patch(patches.Circle((agent_col, agent_row), 0.25, color='lightblue', zorder=17))
    
    # Add goal marker (green star)
    goal_row, goal_col = env.pos_goal
    ax.add_patch(patches.RegularPolygon((goal_col, goal_row), 6, radius=0.4, 
                                       facecolor='green', edgecolor='darkgreen', zorder=15))
    
    # Add start marker (orange square)
    start_row, start_col = (0, 0)
    ax.add_patch(patches.Rectangle((start_col-0.3, start_row-0.3), 0.6, 0.6, 
                                  facecolor='orange', edgecolor='darkorange', zorder=15))
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1.5)
    
    # Add labels and title
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Column", fontsize=12)
    ax.set_ylabel("Row", fontsize=12)
    
    # Add info text
    info_text = f"Size: {env.height}×{env.width}\\nColors: {env.n_colors}\\nObstacles: {env.obstacle_mask.sum()}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
    
    # Add legend
    legend_elements = [
        patches.Patch(color='blue', label='Agent'),
        patches.Patch(color='green', label='Goal'),
        patches.Patch(color='orange', label='Start'),
        patches.Patch(color='black', label='Obstacle'),
        patches.Patch(color='white', label='Floor')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def visualize_all_environments_improved():
    """Visualize all four test environment configurations with improved styling."""
    print("Creating improved visualizations for all environments...")
    
    envs = create_test_environments()
    titles = [
        "Environment 1: 5×5, Unique Colors, No Obstacles",
        "Environment 2: 5×5, 5 Colors, No Obstacles", 
        "Environment 3: 10×10, 7 Colors, 10% Obstacles",
        "Environment 4: 10×10, 4 Colors, 10% Obstacles"
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle("GridWorld Environment Configurations (Improved)", fontsize=20, fontweight='bold')
    
    for i, (env, title) in enumerate(zip(envs, titles)):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Create color map for this environment
        colors = ['white', 'lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 
                  'lightpink', 'lightgray', 'wheat', 'lavender', 'mistyrose'] * 3
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
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.8)
        
        # Add labels
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        
        # Add goal marker (green star)
        goal_row, goal_col = env.pos_goal
        ax.add_patch(patches.RegularPolygon((goal_col, goal_row), 6, radius=0.3, 
                                           facecolor='green', edgecolor='darkgreen', zorder=10))
        
        # Add start marker (orange square)
        start_row, start_col = (0, 0)
        ax.add_patch(patches.Rectangle((start_col-0.25, start_row-0.25), 0.5, 0.5, 
                                      facecolor='orange', edgecolor='darkorange', zorder=10))
        
        # Add info text
        info_text = f"Size: {env.height}×{env.width}\\nColors: {env.n_colors}\\nObstacles: {env.obstacle_mask.sum()}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
        
        env.close()
    
    plt.tight_layout()
    plt.savefig('all_environments_improved.png', dpi=150, bbox_inches='tight')
    print("Saved improved combined visualization to 'all_environments_improved.png'")
    plt.show()


def test_obstacle_generation():
    """Test that obstacle generation creates exactly 10% obstacles."""
    print("Testing obstacle generation...")
    
    for seed in [42, 123, 456]:
        mask = create_random_obstacle_mask(10, 10, 0.1, seed=seed)
        total_cells = 10 * 10
        obstacle_count = mask.sum()
        expected_obstacles = int(total_cells * 0.1)
        
        print(f"Seed {seed}: {obstacle_count}/{total_cells} obstacles ({obstacle_count/total_cells*100:.1f}%)")
        print(f"  Expected: {expected_obstacles} obstacles")
        print(f"  Start position (0,0) is obstacle: {mask[0, 0]}")
        print(f"  Goal position (9,9) is obstacle: {mask[9, 9]}")
        print()


if __name__ == "__main__":
    print("🎨 Improved GridWorld Environment Visualizations")
    print("=" * 60)
    
    # Test obstacle generation
    test_obstacle_generation()
    
    # Visualize all environments
    visualize_all_environments_improved()
    
    print("🎉 Improved visualizations completed!")
