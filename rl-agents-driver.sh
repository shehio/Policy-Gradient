# Train a DQN agent on the CartPole-v0 environment
python3 rl-agents/scripts/experiments.py evaluate configs/CartPoleEnv/env.json configs/CartPoleEnv/DQNAgent.json --train --episodes=200

# Run a benchmark of several agents interacting with environments
python3 rl-agents/scripts/experiments.py benchmark cartpole_benchmark.json --test --processes=4