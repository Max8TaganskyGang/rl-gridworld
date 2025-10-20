# 🎮 GridWorld Reinforcement Learning Environment

A comprehensive implementation of GridWorld environments for Reinforcement Learning research, featuring multiple environment configurations, visualization tools, and support for DQN and PPO algorithms.

## 🌟 Features

- **Multiple Environment Configurations**: 4 different GridWorld setups with varying complexity
- **Gymnasium Compatible**: Full implementation of the Gymnasium interface
- **Rich Visualizations**: Both PNG images and ASCII art for terminal display
- **Extensible Design**: Easy to add new environment variants
- **Comprehensive Testing**: Full test suite for environment validation

## 🏗️ Environment Configurations

| Environment | Size | Colors | Obstacles | Description |
|-------------|------|--------|-----------|-------------|
| Env 1 | 5×5 | 25 | None | Fully observable MDP for debugging |
| Env 2 | 5×5 | 5 | None | Partially observable environment |
| Env 3 | 10×10 | 7 | 10% | Scalability test with obstacles |
| Env 4 | 10×10 | 4 | 10% | High uncertainty with color repetition |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/rl-gridworld.git
cd rl-gridworld

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from gridworld_env import GridWorldEnv, create_test_environments

# Create a simple environment
env = GridWorldEnv(height=5, width=5, n_colors=5)

# Reset and take actions
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

# Create all test environments
envs = create_test_environments()
```

### Visualization

```bash
# Generate PNG visualizations
python visualize_env.py

# ASCII visualization in terminal
python ascii_visualize.py

# Run tests
python test_env.py
```

## 📊 Environment Details

### Observation Space
- **Type**: Discrete
- **Values**: `n_colors + 3` (floor colors + target + wall + obstacle)

### Action Space
- **Type**: Discrete
- **Values**: 4 actions (up, right, down, left)

### Rewards
The environment uses a configurable reward system:

- **Goal reached**: +10.0 (configurable via `goal_reward`)
- **Distance reward**: Closer to goal = higher reward (scaled by `distance_reward_scale`)
- **Step penalty**: -0.01 per step to encourage efficiency (configurable via `step_penalty`)
- **Collision penalty**: -0.1 for hitting obstacles/walls (configurable via `collision_penalty`)

**Example reward configurations:**
- **Sparse**: Only goal reward (hard to learn)
- **Dense**: Balanced progress + efficiency rewards (default)
- **Efficiency**: Strong penalties for wasted steps
- **Exploration**: High rewards for getting closer

## 🎨 Visualization Examples

The project includes comprehensive visualization tools:

- **PNG Images**: High-quality visualizations of all environments
- **ASCII Art**: Terminal-friendly representations
- **Agent Movement**: Step-by-step movement demonstrations

## 📁 Project Structure

```
rl-gridworld/
├── gridworld_env.py      # Main environment implementation
├── visualize_env.py      # PNG visualization script
├── ascii_visualize.py    # ASCII visualization script
├── test_env.py           # Environment testing
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_env.py
```

This will validate:
- Environment initialization
- Action execution
- Observation generation
- Reward calculation
- All four environment configurations

## 🔬 Research Applications

This environment is designed for:
- **Algorithm Development**: Test DQN, PPO, and other RL algorithms
- **Curriculum Learning**: Progressive difficulty across environments
- **Transfer Learning**: Study generalization across different configurations
- **Visualization Research**: Complex observation spaces with MNIST encoding

## 📈 Future Work

- [ ] DQN implementation
- [ ] PPO implementation
- [ ] MNIST observation encoding
- [ ] Environment vectorization
- [ ] Performance comparisons

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with [Gymnasium](https://gymnasium.farama.org/)
- Visualization powered by [Matplotlib](https://matplotlib.org/)
- Inspired by classic GridWorld environments in RL literature
