import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap


class GridWorldEnv(gym.Env):
    """
    GridWorld environment implementing gymnasium interface.
    
    The environment consists of a rectangular grid where an agent must navigate
    from its starting position to a target position, avoiding obstacles.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        height: int = 5,
        width: int = 5,
        n_colors: int = 5,
        obstacle_mask: Optional[np.ndarray] = None,
        pos_goal: Tuple[int, int] = (4, 4),
        pos_agent: Union[Tuple[int, int], np.ndarray] = (0, 0),
        see_obstacle: bool = True,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
        floor_colors: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        # Reward parameters
        goal_reward: float = 10.0,
        step_penalty: float = -0.01,
        collision_penalty: float = -0.1,
        distance_reward_scale: float = 0.1,
    ):
        """
        Initialize GridWorld environment.
        
        Args:
            height: Height of the grid
            width: Width of the grid
            n_colors: Number of different floor colors
            obstacle_mask: Binary mask for obstacles (None for no obstacles)
            pos_goal: Target position (row, col)
            pos_agent: Starting position or probability distribution
            see_obstacle: Whether agent sees obstacle color when hitting it
            max_steps: Maximum steps per episode
            render_mode: Rendering mode
            floor_colors: Predefined floor coloring (None for random)
            seed: Random seed
        """
        super().__init__()
        
        self.height = height
        self.width = width
        self.n_colors = n_colors
        self.pos_goal = pos_goal
        self.pos_agent_init = pos_agent
        self.see_obstacle = see_obstacle
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Reward parameters
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.collision_penalty = collision_penalty
        self.distance_reward_scale = distance_reward_scale
        
        # Set up random number generator
        self.np_random = np.random.RandomState(seed)
        
        # Initialize obstacle mask
        if obstacle_mask is None:
            self.obstacle_mask = np.zeros((height, width), dtype=bool)
        else:
            self.obstacle_mask = obstacle_mask.astype(bool)
        
        # Initialize floor colors
        if floor_colors is None:
            self.floor_colors = self._generate_random_floor_colors()
        else:
            self.floor_colors = floor_colors
        
        # Define action space (4 directions: up, right, down, left)
        self.action_space = spaces.Discrete(4)
        
        # Define observation space (one-hot encoding of cell type)
        # Total colors: n_colors (floor) + 3 (target, wall, obstacle)
        self.observation_space = spaces.Discrete(self.n_colors + 3)
        
        # Initialize state
        self.current_pos = None
        self.step_count = 0
        
        # Action mappings
        self.action_to_direction = {
            0: (-1, 0),  # up
            1: (0, 1),   # right
            2: (1, 0),   # down
            3: (0, -1),  # left
        }
    
    def _generate_random_floor_colors(self) -> np.ndarray:
        """Generate random floor coloring."""
        floor_colors = np.zeros((self.height, self.width), dtype=int)
        
        for i in range(self.height):
            for j in range(self.width):
                if not self.obstacle_mask[i, j] and (i, j) != self.pos_goal:
                    floor_colors[i, j] = self.np_random.randint(0, self.n_colors)
        
        return floor_colors
    
    def _get_cell_type(self, pos: Tuple[int, int]) -> int:
        """Get the type/color of a cell at given position."""
        row, col = pos
        
        # Check bounds
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return self.n_colors + 1  # wall
        
        # Check if it's the goal
        if pos == self.pos_goal:
            return self.n_colors  # target
        
        # Check if it's an obstacle
        if self.obstacle_mask[row, col]:
            return self.n_colors + 2  # obstacle
        
        # Return floor color
        return self.floor_colors[row, col]
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (not obstacle and within bounds)."""
        row, col = pos
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return False
        return not self.obstacle_mask[row, col]
    
    def _sample_start_position(self) -> Tuple[int, int]:
        """Sample starting position based on pos_agent parameter."""
        if isinstance(self.pos_agent_init, tuple):
            return self.pos_agent_init
        
        # pos_agent is a probability distribution
        prob_dist = self.pos_agent_init.copy()
        
        # Apply obstacle mask (set obstacle probabilities to 0)
        prob_dist[self.obstacle_mask] = 0
        
        # Normalize probabilities
        prob_dist = prob_dist / prob_dist.sum()
        
        # Sample position
        flat_idx = self.np_random.choice(prob_dist.size, p=prob_dist.flatten())
        row, col = np.unravel_index(flat_idx, prob_dist.shape)
        
        return (row, col)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[int, dict]:
        """Reset the environment."""
        if seed is not None:
            self.np_random.seed(seed)
        
        # Sample starting position
        self.current_pos = self._sample_start_position()
        self.step_count = 0
        
        # Get initial observation
        observation = self._get_cell_type(self.current_pos)
        
        info = {
            "current_pos": self.current_pos,
            "step_count": self.step_count,
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        """Execute one step in the environment."""
        if self.current_pos is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        # Get direction for action
        direction = self.action_to_direction[action]
        new_pos = (self.current_pos[0] + direction[0], self.current_pos[1] + direction[1])
        
        # Check if new position is valid
        if self._is_valid_position(new_pos):
            self.current_pos = new_pos
            observation = self._get_cell_type(self.current_pos)
        else:
            # Hit obstacle or wall
            if self.see_obstacle:
                observation = self._get_cell_type(new_pos)
            else:
                observation = self._get_cell_type(self.current_pos)
        
        # Calculate reward
        reward = 0.0
        terminated = False
        truncated = False
        
        # Check if reached goal
        if self.current_pos == self.pos_goal:
            reward = self.goal_reward  # Large positive reward for reaching goal
            terminated = True
        else:
            # Distance-based reward (closer to goal = higher reward)
            goal_row, goal_col = self.pos_goal
            current_row, current_col = self.current_pos
            distance_to_goal = abs(current_row - goal_row) + abs(current_col - goal_col)
            max_distance = self.height + self.width - 2  # Manhattan distance from corner to corner
            
            # Normalized distance reward (closer = higher)
            distance_reward = (max_distance - distance_to_goal) / max_distance * self.distance_reward_scale
            
            # Small penalty for each step to encourage efficiency
            step_penalty = self.step_penalty
            
            # Penalty for hitting obstacles/walls
            collision_penalty = 0.0
            if not self._is_valid_position(new_pos):
                collision_penalty = self.collision_penalty
            
            reward = distance_reward + step_penalty + collision_penalty
        
        # Check if max steps reached
        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True
        
        info = {
            "current_pos": self.current_pos,
            "step_count": self.step_count,
            "action": action,
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_human(self):
        """Render environment for human viewing."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create color map
        colors = ['white'] * self.n_colors + ['green', 'red', 'black']  # target, wall, obstacle
        cmap = ListedColormap(colors)
        
        # Create grid visualization
        grid = np.zeros((self.height, self.width))
        
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) == self.pos_goal:
                    grid[i, j] = self.n_colors  # target (green)
                elif self.obstacle_mask[i, j]:
                    grid[i, j] = self.n_colors + 2  # obstacle (black)
                else:
                    grid[i, j] = self.floor_colors[i, j]
        
        # Display grid
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=self.n_colors + 2)
        
        # Add agent position (always visible)
        if self.current_pos is not None:
            agent_row, agent_col = self.current_pos
            # Make agent more visible with a larger circle and border
            ax.add_patch(patches.Circle((agent_col, agent_row), 0.4, color='blue', zorder=10))
            ax.add_patch(patches.Circle((agent_col, agent_row), 0.3, color='lightblue', zorder=11))
        
        # Add goal marker
        goal_row, goal_col = self.pos_goal
        ax.add_patch(patches.Circle((goal_col, goal_row), 0.3, color='green', zorder=10))
        
        # Add start marker
        start_row, start_col = (0, 0)
        ax.add_patch(patches.Circle((start_col, start_row), 0.2, color='orange', zorder=10))
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        
        ax.set_title("GridWorld Environment")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        
        plt.tight_layout()
        plt.show()
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render environment as RGB array."""
        fig, ax = plt.subplots(figsize=(6, 6))
        
        colors = ['white'] * self.n_colors + ['green', 'red', 'black']  # target, wall, obstacle
        cmap = ListedColormap(colors)
        
        grid = np.zeros((self.height, self.width))
        
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) == self.pos_goal:
                    grid[i, j] = self.n_colors
                elif self.obstacle_mask[i, j]:
                    grid[i, j] = self.n_colors + 2
                else:
                    grid[i, j] = self.floor_colors[i, j]
        
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=self.n_colors + 2)
        
        # Add agent position (always visible)
        if self.current_pos is not None:
            agent_row, agent_col = self.current_pos
            ax.add_patch(patches.Circle((agent_col, agent_row), 0.4, color='blue', zorder=10))
            ax.add_patch(patches.Circle((agent_col, agent_row), 0.3, color='lightblue', zorder=11))
        
        # Add goal marker
        goal_row, goal_col = self.pos_goal
        ax.add_patch(patches.Circle((goal_col, goal_row), 0.3, color='green', zorder=10))
        
        # Add start marker
        start_row, start_col = (0, 0)
        ax.add_patch(patches.Circle((start_col, start_row), 0.2, color='orange', zorder=10))
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return buf
    
    def close(self):
        """Close the environment."""
        pass


def create_random_obstacle_mask(height: int, width: int, obstacle_prob: float = 0.1, seed: Optional[int] = None) -> np.ndarray:
    """
    Create a random obstacle mask with exactly the specified percentage of obstacles.
    
    Args:
        height: Height of the grid
        width: Width of the grid
        obstacle_prob: Fraction of cells that should be obstacles (0.0 to 1.0)
        seed: Random seed
    
    Returns:
        Binary mask where True indicates obstacle
    """
    rng = np.random.RandomState(seed)
    
    # Create flat array of all positions
    total_cells = height * width
    num_obstacles = int(total_cells * obstacle_prob)
    
    # Create mask with exactly num_obstacles obstacles
    mask_flat = np.zeros(total_cells, dtype=bool)
    obstacle_indices = rng.choice(total_cells, size=num_obstacles, replace=False)
    mask_flat[obstacle_indices] = True
    
    # Reshape to grid
    mask = mask_flat.reshape(height, width)
    
    # Ensure goal and start positions are not obstacles
    mask[0, 0] = False  # Start position
    mask[height-1, width-1] = False  # Goal position
    
    return mask


def create_test_environments():
    """Create the four test environment configurations."""
    
    # Environment 1: 5x5 grid, unique colors, no obstacles
    env1 = GridWorldEnv(
        height=5,
        width=5,
        n_colors=25,  # Unique color for each position
        obstacle_mask=None,
        pos_goal=(4, 4),
        pos_agent=(0, 0),
        see_obstacle=True,
        max_steps=50,
        seed=42
    )
    
    # Environment 2: 5x5 grid, 5 floor colors, no obstacles
    env2 = GridWorldEnv(
        height=5,
        width=5,
        n_colors=5,
        obstacle_mask=None,
        pos_goal=(4, 4),
        pos_agent=(0, 0),
        see_obstacle=True,
        max_steps=50,
        seed=42
    )
    
    # Environment 3: 10x10 grid, 7 floor colors, 10% obstacles
    obstacle_mask_3 = create_random_obstacle_mask(10, 10, 0.1, seed=42)
    env3 = GridWorldEnv(
        height=10,
        width=10,
        n_colors=7,
        obstacle_mask=obstacle_mask_3,
        pos_goal=(9, 9),
        pos_agent=(0, 0),
        see_obstacle=True,
        max_steps=100,
        seed=42
    )
    
    # Environment 4: 10x10 grid, 4 floor colors, 10% obstacles
    obstacle_mask_4 = create_random_obstacle_mask(10, 10, 0.1, seed=123)
    env4 = GridWorldEnv(
        height=10,
        width=10,
        n_colors=4,
        obstacle_mask=obstacle_mask_4,
        pos_goal=(9, 9),
        pos_agent=(0, 0),
        see_obstacle=True,
        max_steps=100,
        seed=123
    )
    
    return [env1, env2, env3, env4]


if __name__ == "__main__":
    # Test the environment
    env = GridWorldEnv(height=5, width=5, n_colors=5, render_mode="human")
    
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial position: {info['current_pos']}")
    
    # Take some random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: action={action}, obs={obs}, reward={reward}, pos={info['current_pos']}")
        
        if terminated or truncated:
            break
    
    env.render()
    env.close()
