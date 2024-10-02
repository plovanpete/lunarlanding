# Lunar Moon Landing AI Project


https://github.com/user-attachments/assets/18cbef19-60d7-4ff7-8482-f8f70abb28a1



## Overview

This project is a reinforcement learning project that implements a Deep Q-Network (DQN) agent to navigate the Lunar Lander environment provided by OpenAI's Gym. The agent is trained to land a lunar module between two flags on the surface of the moon. This project showcases the use of PyTorch for building neural networks, as well as the Gymnasium library for environment simulation.

## Instructions on how to run it:

First, you'll want to create a virtual environment. Do so by using this command:
```
python -m venv venv
```
Do this command after you created your venv *(Only for Windows)*:
```
./venv/Scripts/activate  
```

Afterwards, install the requirements needed: 
```
pip install -r requirements.txt
```
or

```
pip install -r requirementsVisualizer.txt
```
Afterwards, run the train_dqn.py file to see the Agent generate multiple steps to get the goal of the score! (200.0 is what we set, you can change it to something else in the train_dqn code):
```
python train_dqn.py
```

You can then (if you downloaded the requirementsVisualize.txt) run this code:
```
python visualize_results.py
```

Make sure you have a checkpoint.pth generated before running that command!

## Script Information:

**agent.py**
- This script defines the agent for the reinforcement learning in which it implements a Deep Q-Network(DQN).
- It will initialize the agent and replay memory first, then steps through by taking the best action.
- It will continue to learn and update to exploit the best rewards using the greedy-epilson policy in the act function.

**network.py**
- This script initializes the neural network class to approximate the action-value (Q-Value) functions for the agent.

**replay_memory.py**
- This script implements the replay buffer for storing the experiences in the reinforcement learning project.

**train_dqn**
- This script implements all of the scripts so far (agent.py, network.py, and replay_memory.py). This is where it intializes the reinforcement learning agent, and trains it over a certain amount of episodes depending on the score (2000 is the max in this case). It will then act using the greedy-epsilon policy and update the knowledge through the expeirences saved in the replay buffer.
