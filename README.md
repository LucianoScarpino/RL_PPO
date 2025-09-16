# Proximal Policy Optimization (PPO)

## Overview

This repository implements the **Proximal Policy Optimization (PPO)** algorithm for continuous control tasks using **MuJoCo** environments.  
The agent optimizes a clipped surrogate objective that allows for multiple epochs of minibatch updates while avoiding destructive policy updates.  
A custom version of the MuJoCo Hopper environment is also included for testing and experimentation.

---

## How to Run

### 1. Initialize Virtual Environment

Set up a Python virtual environment and install the required dependencies:

```bash
python3 -m venv <path_to_venv>
source <path_to_venv>/bin/activate   # Linux/Mac
# <path_to_venv>\Scripts\activate    # Windows
pip install -r requirements.txt
```

---

## Run the Training

Execute the main script to train the agent:
```bash
python main.py
```

In addition, you can specify the following arguments:
- **-–train_domain** : training domain, values source or target (default: source)
– **--test_domain** : testing domain, values source or target (default: target)
- **–-learning_rate** : learning rate for optimization (default: 3e-4)
– **--batch_size** : batch size for training (default: 64)
– **--buffer_size** : replay buffer size (default: 1e5)
– **--n_episodes** : number of training episodes (default: 100000)
- **--fc1_dim** : number of units in the first hidden layer (default: 128)
- **--fc2_dim** : number of units in the second hidden layer (default: 128)

Here is an example of full script:
```bash
python main.py --n_episodes 500 --batch_size 256 --learning_rate 0.0001
```

---

## Requirements
The project depends on the following main packages:

```bash
torch
numpy
gymnasium
mujoco
```
Python 3.9+ is recommended.

---

## Implementation

## Algorithm

The PPO agent is implemented with the following components:
1. **Policy Network** (Actor)
  	- Outputs a stochastic Gaussian policy for continuous action spaces.
	- Optimized using the PPO clipped surrogate objective with entropy regularization.
2.	**Value Network** (Critic)
	- Estimates the state-value function.
	- Trained to minimize the mean squared error between predicted and target returns.
3.	**Replay Buffer / Trajectory Storage**
	- Stores trajectories of interactions.
	- Computes advantages using Generalized Advantage Estimation (GAE).
4.	**Environment**
	- Implemented using MuJoCo.
	- Includes a custom Hopper environment for experimentation.
