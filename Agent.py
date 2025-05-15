from Networks import PPOmemory,ActorNetwork,CriticNetwork

import numpy as np
import torch as T

class Agent():
    def __init__(self,n_actions,input_dims,gamma=0.99,alpha= 0.0003,g_lambda=0.95,
                 policy_clip=0.2,batch_size=64,N=2048,n_epochs=10):
        
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.g_lambda = g_lambda

        self.actor = ActorNetwork(n_actions,input_dims,alpha)
        self.critic = CriticNetwork(input_dims,alpha)
        self.memory = PPOmemory(batch_size)

    def remember(self,state,action,probs,vals,reward,done):
        self.memory.store_memory(state,action,probs,vals,reward,done)
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_chechkpoint()
        print('...Model Saved...')

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_chechpoint()
        print('...Model Loaded...')

    def choose_action(self,observation):
        state = T.tensor(observation, dtype=T.float32).unsqueeze(0).to(self.actor.device)

        dist = self.actor(state)                                                    #Equivalent to call forward
        value = self.critic(state)
        action = dist.sample()                                                      #Don't care about gradient

        probs = dist.log_prob(action).sum(axis=-1).item()
        action = np.array(action.cpu().detach().numpy(), dtype=np.float32).reshape(-1)
        value = value.item()

        return probs, action, value
    
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr,\
            reward_arr, done_arr, batches = \
                self.memory.generate_batches()
            
        values = vals_arr
        values = np.append(values, 0.0)
        advantage = np.zeros(len(reward_arr),dtype=np.float32)

        for t in range(len(reward_arr)-1):
            discount = 1                                        #Initialize dicount factor (lambda_new) as one, next discount by gamma * lambda
                                                                #The lambda parameter is a smooth factor  
            a_t = 0
            for k in range(t,len(reward_arr)-1):
                a_t += discount * (reward_arr[k] + self.gamma * values[k+1] * (1 - int(done_arr[k])) - values[k])
                discount *= self.gamma*self.g_lambda
            advantage[t] = a_t
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)
        advantage = T.tensor(advantage).to(self.actor.device)

        for batch in batches:
            states = T.tensor(state_arr[batch],dtype=T.float32).to(self.actor.device)
            old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
            actions = T.tensor(np.array(action_arr[batch]), dtype=T.float32).to(self.actor.device)


            #Implementation of PPO
            dist = self.actor(states)
            critic_value = self.critic(states)

            critic_value = critic_value.view(-1)

            new_probs = dist.log_prob(actions).sum(axis=-1)
            prob_ratio = new_probs.exp()/old_probs.exp()
            weighted_probs = advantage[batch]*prob_ratio                                                            #Adjust importance of advantage
            weighted_clipped_probs = T.clamp(prob_ratio,1-self.policy_clip,1+self.policy_clip)*advantage[batch]     #clips prob_ratio into (1-eps,1+eps)
            actor_loss = - T.min(weighted_probs,weighted_clipped_probs).mean()                                      #Takes the minimum, see next comment
            #Aim is to have smalles loss as possible since it involves into having small updates of weights
            #Indeed high update of weights by optimizer could cause drop of learning (in terms of reward)

            returns = advantage[batch] + values[batch]
            critic_loss = (returns - critic_value)**2
            critic_loss = critic_loss.mean()

            total_loss = actor_loss + 0.5 * critic_loss                 #Give more emphasy to actor_loss
            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            total_loss.backward()
            self.actor.optimizer.step()
            self.critic.optimizer.step()
    
        self.memory.clear_memory()
