🧠 Deep Q-Network (DQN) — Grid World Agent
  - This project implements a Deep Q-Network (DQN) from scratch to train an agent in a custom Grid World environment. The goal of the agent is to learn the optimal path through trial and error, using rewards and penalties to shape its behavior.
  - The implementation demonstrates the fundamentals of reinforcement learning and how deep learning can be applied to solve sequential decision-making problems.

🚀 Project Overview
  - Environment: Custom Grid World (discrete states & actions)
  - Algorithm: Deep Q-Network (DQN) — off-policy, value-based method
  - Objective: Train the agent to find the correct/shortest path to the goal
  - Learning Paradigm: Agent improves over episodes by balancing exploration vs. exploitation

⚙️ Key Features
  - DQN implementation built from scratch (no pre-built RL libraries)
  - Experience replay buffer for stable training
  - Target network to reduce Q-value oscillations
  - Epsilon-greedy strategy for exploration
  - Training metrics (reward curves, loss) to visualize learning progress
