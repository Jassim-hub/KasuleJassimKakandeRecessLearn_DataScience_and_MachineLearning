# Reinforcement learning(RL)
# What is RL?
# Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward.
# Key components of RL:
# 1. Agent: The learner or decision maker.
# 2. Environment: The external system with which the agent interacts.
# 3. Action: The set of all possible moves the agent can make.
# 4. State: The current situation of the agent in the environment.
# 5. Reward: The feedback signal received after taking an action in a state.
# 6. Policy: The strategy that the agent employs to determine its actions based on the current state.
# 7. Value Function: A function that estimates the expected cumulative reward from a given state or state-action pair.

# Key characteristics of RL:
# No labeled data: Unlike supervised learning, RL does not require labeled input-output pairs.
# Focus on larning from interaction: The agent learns by interacting with the environment and receiving feedback in the form of rewards.
# Involves exploration and exploitation: The agent must balance exploring new actions to discover their rewards and exploiting known actions that yield high rewards.
# Work through delay rewards:The agent may not receive rewards after a series of actions,requiring it to learn long-term strategies.

# Reinforcement learning algorithms:
# 1. Q-Learning: A model-free algorithm that learns the value of actions in states to derive an optimal policy.
# 2. Deep Q-Networks (DQN): Combines Q-learning with deep neural networks to handle high-dimensional state spaces.
# 3. Policy Gradient Methods: Directly optimize the policy by adjusting the parameters to maximize expected rewards.
# 4.Actor-Critic Methods: Combines value-based and policy-based approaches, using an actor to select actions and a critic to evaluate them.
# 5.Proximal Policy Optimization:
# 6.Trust Region Policy Optimization(TRPO):
# 7.Monte Carlo Methods:

# Q-Learning:
# Environment (Position,Goal,Reward)

# Action (Move Left, Move Right, Stay)
# State(Current Position)
# Reward(Positive for reaching the goal,negative for hitting a wall)

# Create a simple q-learning agent to navigate a grid environment while avoiding walls
import numpy as np
import random

# Define environment parameters
episodes = 1000  # Number of episodes for trainind
learning_rate = 0.8  # Learning rate
gamma = 0.9  # Discount factor for future rewards
episilon = 0.3  # Exploration rate(Probabilty of taking a random action)


# training loop
for episode in range(episodes):
    state = random.randint(
        0, position - 1
    )  # Randomly select an initial state(position)

# Action selection(Episilon-greedy policy)
if random.uniform(0, 1) < episilon:  # Explore with probability episilon
    action = random.randint(0, actions - 1)

# take action
if action == 0:  # Move left
    next_state = max(0, state - 1)
elif action == 1:  # Move right
    next_state = min(position - 1, state + 1)
# Assignment train RL agent to navigate to cross the road with action right,left and
