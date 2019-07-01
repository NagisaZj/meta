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

from bandit import bandit

from metalearner import network

from sampler import sampler






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
    batch_sampler = sampler(args.batch_size,args.num_bandits)
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
    parser.add_argument('--outer_lr', help='lr for outer update', type=float, default=0.3)
    parser.add_argument('--first_order', help='use first order approximation', action='store_true')
    parser.add_argument('--plot', help='whether plot figure', action='store_true')
    parser.add_argument('--dir', help='log directory', type=str,default='./data/')
    parser.add_argument('--number', help='number for storing data', type=int, default=5)
    parser.add_argument('--batch_size', help='batch size(also episode length)', type=int, default=64)
    parser.add_argument('--max_kl', help='max kl divergence', type=float, default= 1e-2)
    parser.add_argument('--cg_iters', type=int, default=10)
    parser.add_argument('--inner_loop_cnt', type=int, default=5)
    parser.add_argument('--cg_damping', help='positive reward for good bandits', type=float, default=1e-5)
    parser.add_argument('--ls_max_steps', help='negative reward for bad bandits', type=int, default=1)
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