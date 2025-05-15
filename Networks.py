import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class PPOmemory:
    def __init__(self,batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0,n_states,self.batch_size)                 #Values go from 0 to n_states and array will contain batch_size elements
                                                                            #This function is made up to take strarting index in the batch to create mini-batch
        indices = np.arange(n_states,dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches
    
    def store_memory(self,state,action,probs,vals,reward,done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

class ActorNetwork(nn.Module):
    def __init__(self,n_actions,input_dims,alpha,fc1_dims=256,fc2_dims=256,chkpt_dir='tmp/ppo'):
        super(ActorNetwork,self).__init__()

        self.checkpoit_file = os.path.join(chkpt_dir,'actor_ppo')                   #it joins(directory,filename) creating the flow (output will be in tmp/ppo/actor_ppo)
        self.actor = nn.Sequential(
            nn.Linear(*input_dims,fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims,fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims,n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        dist = self.actor(state)                #Actor observe state (neural network input) and creates a distribution of action probabilities
        dist = Categorical(dist)

        return dist
    
    def save_checkpoint(self):
        T.save(self.state_dict(),self.checkpoit_file)       #Save weights (parameters) into directory file
    
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoit_file))   #Upload weights saved before

class CriticNetwork(nn.Module):
    def __init__(self,input_dims,alpha,fc1_dims=256,fc2_dims=256,chkpt_dir='tmp/ppo'):
        super(CriticNetwork,self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir,'critic_ppo')
        self.critic = nn.Sequential(
            nn.Linear(*input_dims,fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims,fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims,1)
        )

        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        value = self.critic(state)              #In this case I'm using a critic that return the V-value estimated (no action is taken into account)

        return value
    
    def save_chechkpoint(self):
        T.save(self.state_dict(),self.checkpoint_file)

    def load_chechpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))