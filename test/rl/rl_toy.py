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

def detach_distribution(pi):
    if isinstance(pi, Categorical):
        distribution = Categorical(logits=pi.logits.detach())
    elif isinstance(pi, Normal):
        distribution = Normal(loc=pi.loc.detach(), scale=pi.scale.detach())
    else:
        raise NotImplementedError('Only `Categorical` and `Normal` '
                                  'policies are valid policies.')
    return distribution

def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.clone().detach()
    r = b.clone().detach()
    x = torch.zeros_like(b).float()
    rdotr = torch.dot(r, r)

    for i in range(cg_iters):
        z = f_Ax(p).detach()
        v = rdotr / torch.dot(p, z)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr.item() < residual_tol:
            break

    return x.detach()

def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        #module.bias.data.zero_()

def weight_init_bias(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()


class bandit():
    def __init__(self,args,num_good):
        self.num_bandits = args.num_bandits
        self.num_good = num_good
        self.good_reward = args.r_positive
        self.bad_reward = args.r_negative
        self.variance = args.variance

    def step(self,action):
        if action == self.num_good:
            reward = np.random.normal(self.good_reward,self.variance,(1,1))
        else:
            reward = np.random.normal(self.bad_reward, self.variance, (1, 1))
        state = np.array([[1]],dtype=np.float32)
        return state,reward,state

    def reset(self):
        state = np.array([[1]], dtype=np.float32)
        return state

class sampler():
    def __init__(self,num_steps):
        self.num_steps = num_steps
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

class episode():
    def __init__(self,s,a,r):
        self.s = s
        self.a = a
        self.r = r

class network(nn.Module):
    def __init__(self,num_layers,num_hidden,num_out):
        super(network,self).__init__()
        self.add_module('layer0', nn.Linear(1, num_hidden))
        for i in range(1,num_layers-1):
            self.add_module('layer{0}'.format(i),nn.Linear(num_hidden,num_hidden))
        self.add_module('layer{0}'.format(num_layers-1), nn.Linear(num_hidden, num_out))
        self.apply(weight_init_bias)
        self.num_layers = num_layers

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = input
        for i in range(self.num_layers):
            output = F.linear(output,
                              weight=params['layer{0}.weight'.format(i)],
                              bias= params['layer{0}.bias'.format(i)])
            if i !=self.num_layers-1:
                output = F.relu(output)
        output = F.softmax(output)
        return output

    def sample_action(self,state,params=None):
        output = self.forward(state,params)
        action_distribution = Categorical(output)
        action = action_distribution.sample()
        return action

    def cal_loss(self,s,a,r,params=None):
        output = self.forward(s,params)
        action_distribution = Categorical(output)
        log_prob = action_distribution.log_prob(a)
        loss = -1*log_prob * r
        return torch.mean(loss)

    def surrogate_loss(self,episodes,inner_losses,old_pis=None):
        losses, kls, pis = [], [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for valid_episodes, old_pi,loss in zip(episodes, old_pis,inner_losses):
            params=self.update_params(loss)
            with torch.set_grad_enabled(old_pi is None):
                pi = Categorical(self.forward(valid_episodes.s, params=params))
                pis.append(detach_distribution(pi))

                if old_pi is None:
                    old_pi = detach_distribution(pi)
                advantages = valid_episodes.r
                log_ratio = (pi.log_prob(valid_episodes.a)
                             - old_pi.log_prob(valid_episodes.a))
                if log_ratio.dim() > 2:
                    log_ratio = torch.sum(log_ratio, dim=2)
                ratio = torch.exp(log_ratio)

                loss = -torch.mean(ratio * advantages)
                losses.append(loss)

                kl = torch.mean(kl_divergence(pi, old_pi))
                kls.append(kl)

        return (torch.mean(torch.stack(losses, dim=0)),
                torch.mean(torch.stack(kls, dim=0)), pis)

    def hessian_vector_product(self, episodes, inner_losses,damping=1e-2):
        """Hessian-vector product, based on the Perlmutter method."""

        def _product(vector):
            kl = self.kl_divergence(episodes,inner_losses)
            grads = torch.autograd.grad(kl, self.parameters(),retain_graph=True,
                                        create_graph=True)
            flat_grad_kl = parameters_to_vector(grads)

            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.parameters(),retain_graph=True)
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector

        return _product

    def kl_divergence(self, episodes, inner_losses,old_pis=None):
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for valid_episodes, old_pi,loss in zip(episodes, old_pis,inner_losses):
            params = self.update_params(loss)

            pi = Categorical(self.forward(valid_episodes.s, params=params))

            if old_pi is None:
                old_pi = detach_distribution(pi)


            kl = torch.mean(kl_divergence(pi, old_pi), dim=0)
            kls.append(kl)

        return torch.mean(torch.stack(kls, dim=0))

    def update_params(self, loss, step_size=0.5, first_order=False):
        """Apply one step of gradient descent on the loss function `loss`, with
        step-size `step_size`, and returns the updated parameters of the neural
        network.
        """
        grads = torch.autograd.grad(loss, self.parameters(),
                                    create_graph=not first_order)
        updated_params = OrderedDict()
        for (name, param), grad in zip(self.named_parameters(), grads):
            updated_params[name] = param - step_size * grad
        return updated_params

    def update_params_clip(self, loss, step_size=0.5, first_order=False):
        """Apply one step of gradient descent on the loss function `loss`, with
        step-size `step_size`, and returns the updated parameters of the neural
        network.
        """
        grads = torch.autograd.grad(loss, self.parameters(),
                                    create_graph=not first_order)
        for g in grads:
            g = torch.clamp(g, -0.5, 0.5)
        updated_params = OrderedDict()
        for (name, param), grad in zip(self.named_parameters(), grads):
            updated_params[name] = param - step_size * grad
        return updated_params

def tttest(learner,args,train_envs, test_envs,log_dir):
    batch_sampler = sampler(args.batch_size)
    for i in range(args.num_updates):
        loss_outer = []
        rew_rem = []
        for j in range(1):
            s, a, r = batch_sampler.sample(train_envs[j], learner)
            inner_loss = learner.cal_loss(s, a, r)
            grads = torch.autograd.grad(inner_loss, learner.parameters())
            grads = parameters_to_vector(grads)
            old_params = parameters_to_vector(learner.parameters())
            vector_to_parameters(old_params - args.outer_lr * grads, learner.parameters())
            mean_rew = torch.mean(r).data.numpy()
            rew_rem.append(mean_rew)

        print(np.mean(rew_rem))

def test(learner,args,train_envs, test_envs,log_dir):
    learner_test = network(args.num_layers, args.num_hidden, args.num_bandits)
    batch_sampler = sampler(args.batch_size)
    max_kl = args.max_kl
    cg_iters = args.cg_iters
    cg_damping = args.cg_damping
    ls_max_steps = args.ls_max_steps
    ls_backtrack_ratio = args.ls_backtrack_ratio
    train_rew = []
    for i in range(args.num_updates):
        #print(i)
        adapt_params=[]
        inner_losses=[]
        adapt_episodes=[]
        rew_rem=[]
        for j in range(args.num_tasks_train):
            e = batch_sampler.sample(train_envs[j],learner)
            inner_loss = learner.cal_loss(e.s,e.a,e.r)
            params = learner.update_params(inner_loss,args.inner_lr,args.first_order)
            a_e = batch_sampler.sample(train_envs[j], learner,params)
            adapt_params.append(params)
            adapt_episodes.append(a_e)
            inner_losses.append(inner_loss)
            mean_rew = torch.mean(a_e.r).data.numpy()
            rew_rem.append(mean_rew)

        print(np.mean(rew_rem))
        train_rew.append(np.mean(rew_rem))
        old_loss, _, old_pis = learner.surrogate_loss(adapt_episodes,inner_losses)
        grads = torch.autograd.grad(old_loss, learner.parameters(),retain_graph=True)
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        hessian_vector_product = learner.hessian_vector_product(adapt_episodes,inner_losses,
                                                             damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads,
                                     cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(learner.parameters())

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step,
                                 learner.parameters())
            loss, kl, _ = learner.surrogate_loss(adapt_episodes, inner_losses,old_pis=old_pis)
            improve = loss - old_loss
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, learner.parameters())

        if (i+1)%10==0:
            test_input = torch.FloatTensor([[1]])
            test_output = learner.forward(test_input).data.numpy()[0]
            plt.figure()
            plt.bar(np.arange(len(test_output)), test_output)
            plt.savefig(log_dir + 'figures/before%i.png' % i)
            plt.close()
            for j in range(args.num_tasks_train):
                test_output = learner.forward(test_input,adapt_params[j]).data.numpy()[0]
                plt.figure()
                plt.bar(np.arange(len(test_output)), test_output)
                plt.savefig(log_dir + 'figures/after%i_%i.png' % (j,i))
                plt.close()


    np.save(log_dir + 'train_rew'+str(args.inner_lr)+'.npy', train_rew)
    plt.figure()
    plt.plot(train_rew)
    plt.show()
    plt.figure()
    plt.plot(train_rew)
    plt.savefig(log_dir+'train_rew.png')

    return

def main(args,log_dir):
    learner = network(args.num_layers,args.num_hidden,args.num_bandits)
    train_envs = []
    test_envs = []
    for i in range(args.num_tasks_train):
        train_envs.append(bandit(args,i))
    for i in range(args.num_tasks_test):
        good_num = np.random.randint(0,args.num_bandits)
        test_envs.append(bandit(args,good_num))
    #learn(learner,args,train_envs,test_envs,log_dir)
    test(learner, args, train_envs, test_envs, log_dir)
        #save(args)

    return






if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_updates', help='number of updates', type=int, default=100)
    parser.add_argument('--num_layers', help='number of layers', type=int, default=3)
    parser.add_argument('--num_hidden', help='number of hidden nodes', type=int, default=10)
    parser.add_argument('--num_bandits',help='the number of bandits',type=int,default=3)
    parser.add_argument('--num_tasks_train', help='the number of meta-training set', type=int, default=3)
    parser.add_argument('--num_tasks_test', help='the number of meta-testing set', type=int, default=5)
    parser.add_argument('--r_positive', help='positive reward for good bandits', type=float, default=1)
    parser.add_argument('--r_negative', help='negative reward for bad bandits', type=float, default=-1)
    parser.add_argument('--variance', help='std for reward sampling', type=float, default=0.1)
    parser.add_argument('--inner_lr', help='lr for inner update', type=float, default=0.5)
    parser.add_argument('--outer_lr', help='lr for outer update', type=float, default=0.1)
    parser.add_argument('--first_order', help='use first order approximation', action='store_true')
    parser.add_argument('--plot', help='whether plot figure', action='store_true')
    parser.add_argument('--dir', help='log directory', type=str,default='./data/')
    parser.add_argument('--number', help='number for storing data', type=int, default=5)
    parser.add_argument('--batch_size', help='batch size(also episode length)', type=int, default=64)
    parser.add_argument('--max_kl', help='max kl divergence', type=float, default= 1e-2)
    parser.add_argument('--cg_iters', type=int, default=10)
    parser.add_argument('--inner_loop_cnt', type=int, default=5)
    parser.add_argument('--cg_damping', help='positive reward for good bandits', type=float, default=1e-5)
    parser.add_argument('--ls_max_steps', help='negative reward for bad bandits', type=int, default=15)
    parser.add_argument('--ls_backtrack_ratio', type=float, default=0.8)
    args = parser.parse_args()
    random.seed(0)
    log_dir=args.dir+str(args.num_layers)+'_'+str(2)+'_'+str(args.inner_lr)+'_'+str(
        args.outer_lr)+'_'+str(args.num_bandits)+'_'+str(args.r_positive)+'_'+str(
        args.r_negative)+'_'+str(args.batch_size)+'_'+str(args.variance)+'ttt/'
    log_dir = args.dir  + str(args.inner_lr) + '_' + str(
        args.outer_lr) + '_' + str(args.num_bandits)+ '_' + str(args.num_tasks_train) + '_' + str(
        args.batch_size) + '_' + str(args.variance) + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(log_dir+'figures/'):
        os.makedirs(log_dir+'figures/')

    main(args,log_dir)