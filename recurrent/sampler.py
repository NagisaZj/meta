import numpy as np
import torch
import argparse
import os
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)

from torch.distributions.kl import kl_divergence
import matplotlib.pyplot as plt
import random
from torch.distributions import Categorical, Normal
from utils import detach_distribution,conjugate_gradient,weight_init,episode

class sampler():
    def __init__(self,num_steps,num_bandits):
        self.num_steps = num_steps
        self.num_bandits = num_bandits
        return

    def sample(self,env,policy,params=None):
        state_batch = np.zeros((self.num_steps,1),dtype=np.float32)
        action_batch = np.zeros((self.num_steps,1),dtype=np.float32)
        reward_batch = np.zeros((self.num_steps,1),dtype=np.float32)
        state = env.reset()
        for i in range(self.num_steps):
            state_batch[i]=state
            action = policy.sample_action(torch.FloatTensor(state),params).data.numpy()[0]
            _,reward,state = env.step(action)
            action_batch[i]=action
            reward_batch[i]=reward
        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        return episode(state_batch,action_batch,reward_batch)

    def balance_sample(self,env,policy,params=None):
        state_batch = np.zeros((self.num_steps,1),dtype=np.float32)
        action_batch = np.zeros((self.num_steps,1),dtype=np.float32)
        reward_batch = np.zeros((self.num_steps,1),dtype=np.float32)
        state = env.reset()
        for i in range(self.num_steps):
            state_batch[i]=state
            #action = policy.sample_action(torch.FloatTensor(state),params).data.numpy()[0]
            action = i % self.num_bandits
            _,reward,state = env.step(action)
            action_batch[i]=action
            reward_batch[i]=reward
        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        return episode(state_batch,action_batch,reward_batch)


class rec_sampler(nn.Module):
    def __init__(self,state_len,action_len,num_hidden=10,num_layers=1):
        super(rec_sampler, self).__init__()
        self.gru = nn.GRU(state_len+1,num_hidden,num_layers)
        self.linear = nn.Linear(num_hidden,action_len)
        self.critic = nn.Linear(num_hidden,1)
        self.optimizer = torch.optim.Adam(self.parameters(), 7e-3)

    def single_forward(self,s,a_prim,h0):
        input = torch.cat((s,a_prim),1)
        input = torch.unsqueeze(input,0)
        #print(input.shape,h0.shape)
        output,hn = self.gru(input,h0)
        action_output = self.linear(torch.squeeze(output,0))
        action_output = F.softmax(action_output)

        return action_output, hn

    def choose_action(self,s,a_prim,h0):
        action_output, hn = self.single_forward(s,a_prim,h0)
        action_distribution = Categorical(action_output)
        action = action_distribution.sample()
        return action.float(),hn,action_distribution


    def cal_loss(self,trajectory,action,rewards,hidden_size):
        h0 = torch.zeros(1,1,hidden_size)
        output,hn = self.gru(trajectory,h0)
        loss = []

        for i in range(output.shape[0]):
            action_output = self.linear(output[i])
            value_output = self.critic(output[i]).detach()
            action_output = F.softmax(action_output)
            action_distribution = Categorical(action_output)
            log_prob = action_distribution.log_prob(action[i])
            loss.append(-1*log_prob*(rewards[i]-value_output))

        value = self.critic(torch.squeeze(output,1))
        c_loss = (value-rewards).pow(2).mean()
        return torch.mean(torch.stack(loss))+c_loss


    def evaluate(self,trajectory,action):
        h0 = torch.zeros(1, 1, hidden_size)
        output, hn = self.gru(trajectory, h0)
        action_output = self.linear(torch.squeeze(output, 1))
        value = self.critic(torch.squeeze(output, 1))
        action_output = F.softmax(action_output)
        action_distribution = Categorical(action_output)
        log_prob = action_distribution.log_prob(action)
        return log_prob,value

    def optimize(self,trajectory,action,rewards,hidden_size):
        h0 = torch.zeros(1,1,hidden_size)
        output,hn = self.gru(trajectory,h0)
        action_output = self.linear(torch.squeeze(output,1))
        action_output = F.softmax(action_output)
        action_distribution = Categorical(action_output)
        log_prob = action_distribution.log_prob(action)
        old_logprob = log_prob.detach()


        for _ in range(3):
            logprob,v = self.evaluate(trajectory,action)
            ratios = torch.exp(logprob - old_logprob.detach())
            advantages = rewards - v.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - 0.4, 1 + 0.4) * advantages
            mse = nn.MSELoss()
            loss = -torch.min(surr1, surr2) + 0.2 * mse(v, rewards)

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()



def cal_rews(action_rem,num_bandits,num_samples):
    gamma = 0
    rem = []
    rew = torch.zeros(num_samples,1)
    for i in range(num_samples):
        if action_rem[i,0]<int(num_bandits/2) and not(action_rem[i,0] in rem):
            rew[i,0]=1
        else:
            rew[i,0]=0
        rem.append(action_rem[i,0])
    rew_decay = torch.zeros(num_samples,1)
    tmp = 0
    for i in range(num_samples-1,-1,-1):
        rew_decay[i,0] = rew[i] + gamma * tmp
        tmp = rew_decay[i,0]
    return rew_decay,rew


if __name__=="__main__":
    num_bandits = 20
    num_samples = int(num_bandits/2)
    hidden_size = num_bandits * 2
    sam = rec_sampler(1,num_bandits,hidden_size)
    rew_mem = []
    rew_mem_sto = []
    rew_mem_good = []
    if os.path.exists('./data') is None:
        os.makedirs('./data')



    #tra_buffer =
    for episodes in range(7000):
        tra_rem = torch.zeros(num_samples, 1, 2)
        action_rem = torch.zeros(num_samples, 1)
        s = torch.FloatTensor([[1]])
        a_prim = torch.FloatTensor([[-1]])
        h0 = torch.zeros(1, 1, hidden_size)
        for i in range(num_samples):
            tra_rem[i, :, :] = torch.cat((s, a_prim), 1).detach()
            a, hn, action_distribution = sam.choose_action(s, a_prim, h0)
            h0 = hn
            a_prim = torch.unsqueeze(torch.FloatTensor(a), 0).detach()
            action_rem[i, :] = a_prim.detach()
            '''loss = torch.mean(action_distribution.log_prob(a))
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            print(i, a, hn, action_distribution.probs)'''
        reward,r = cal_rews(action_rem,num_bandits,num_samples)
        sam.optimize(tra_rem,action_rem,reward,hidden_size)




        if (episodes+1)%30 ==0:
            action_sto = np.random.randint(0, num_bandits, (num_samples, 1))
            sto_r, _ = cal_rews(action_sto, num_bandits, num_samples)

            action_good = np.random.randint(0, int(num_bandits / 2), (num_samples, 1))
            good_r, _ = cal_rews(action_good, num_bandits, num_samples)
            rew_mem.append(torch.sum(r).data.numpy())
            rew_mem_sto.append(torch.sum(sto_r).data.numpy())
            rew_mem_good.append(torch.sum(good_r).data.numpy())
            print(episodes,rew_mem[-1],rew_mem_sto[-1],rew_mem_good[-1])
            np.save('rew.npy',rew_mem)
            np.save('rew_sto.npy', rew_mem)
            np.save('rew_good.npy', rew_mem)

    plt.figure()
    plt.plot(rew_mem,label='rnn policy',color='b')
    plt.plot(rew_mem_sto, label='random policy', color='r')
    plt.plot(rew_mem_good, label='random policy(with known target distribution)', color='y')
    plt.legend()
    plt.show()

