import numpy as np
import torch
import argparse
import os
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
import matplotlib.pyplot as plt

def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        #module.bias.data.zero_()

def weight_init_bias(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

class network(nn.Module):
    def __init__(self,num_layers,num_hidden):
        super(network,self).__init__()
        self.add_module('layer0', nn.Linear(1, num_hidden))
        for i in range(1,num_layers-1):
            self.add_module('layer{0}'.format(i),nn.Linear(num_hidden,num_hidden))
        self.add_module('layer{0}'.format(num_layers-1), nn.Linear(num_hidden, 1))
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

        return output

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

def learn(learner,args,task_target,log_dir):
    learner_test = network(args.num_layers,args.num_hidden)
    loss_rem = []
    loss_rem_test = []
    max_grads=[]
    for i in range(3):
        loss_rem_test.append([])
    for update in range(args.num_updates):

        input = torch.FloatTensor(np.random.normal(0,1,(args.batch_size,1)))
        lossfun = nn.MSELoss()
        losses = []
        for i in range(3):
            mean = i - 1
            input_tem = torch.FloatTensor(np.random.normal(mean*2,1,(args.batch_size,1)))
            target_tem = input_tem * args.B3
            params=None
            old_params = parameters_to_vector(learner.parameters())
            vector_to_parameters(old_params, learner_test.parameters())
            for _ in range(3):
                loss = lossfun(learner_test.forward(input_tem), target_tem)
                grads = torch.autograd.grad(loss, learner_test.parameters())
                grads = parameters_to_vector(grads)
                grads = torch.clamp(grads, -0.5, 0.5)
                old_params = parameters_to_vector(learner_test.parameters())
                vector_to_parameters(old_params - args.outer_lr * grads, learner_test.parameters())

            loss_tem=lossfun(learner_test.forward(input_tem), target_tem)
            print(i-1,loss_tem)
            loss_rem_test[i].append(loss_tem.data.numpy())


        for i in range(2):
            #input = torch.FloatTensor(np.random.normal((i-1)*1, 1, (args.batch_size, 1)))
            target = input * task_target[i]
            loss = lossfun(learner.forward(input), target)
            params = learner.update_params(loss, step_size=args.inner_lr, first_order=args.first_order)
            losses.append(lossfun(learner.forward(input, params), target))
            #if i==0:
            #    losses.append(loss)

        total_loss = torch.mean(torch.stack(losses, dim=0))
        loss_rem.append(total_loss.data.numpy())
        grads = torch.autograd.grad(total_loss, learner.parameters())
        grads = parameters_to_vector(grads)
        #grads = torch.clamp(grads,-0.5,0.5)
        max_grads.append(torch.max(torch.abs(grads)).data.numpy())
        old_params = parameters_to_vector(learner.parameters())
        vector_to_parameters(old_params - args.outer_lr * grads, learner.parameters())

        print('loss', total_loss.data.numpy())
    np.save(log_dir + 'loss4-%i.npy'%args.number , loss_rem)
    np.save(log_dir + 'grads-%i.npy' % args.number, max_grads)
    for i in range(3):
        np.save(log_dir + 'test%i.npy'%i, loss_rem_test[i])

    plt.figure()
    plt.plot(loss_rem,label='train loss(input mean = 0)')
    plt.legend()
    plt.figure()
    plt.plot(loss_rem_test[0], label='test loss (input mean = -1)')
    plt.legend()
    plt.figure()
    plt.plot(loss_rem_test[1], label='test loss (input mean = 0)')
    plt.legend()
    plt.figure()
    plt.plot(loss_rem_test[2], label='test loss (input mean = 1)')
    plt.legend()

    plt.figure()
    plt.plot(loss_rem, label='train loss(input mean = 0)')
    plt.plot(loss_rem_test[0], label='test loss (input mean = -1)')
    plt.plot(loss_rem_test[1], label='test loss (input mean = 0)')
    plt.plot(loss_rem_test[2], label='test loss (input mean = 1)')
    plt.legend()
    plt.figure()
    plt.plot(loss_rem, label='train loss(input mean = 0)')
    plt.plot(loss_rem_test[1], label='test loss (input mean = 0)')
    plt.legend()
    plt.figure()
    plt.plot(max_grads)
    plt.show()
    return



def main(args,log_dir):
    learner = network(args.num_layers,args.num_hidden)
    task_target = [args.B1, args.B2, args.B3]
    learn(learner,args,task_target,log_dir)
        #save(args)

    return






if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_updates', help='number of updates', type=int, default=1000)
    parser.add_argument('--num_layers', help='number of layers', type=int, default=3)
    parser.add_argument('--num_hidden', help='number of hidden nodes', type=int, default=15)
    parser.add_argument('--B1',help='the target of regression task 1',type=int,default=3)
    parser.add_argument('--B2', help='the target of regression task 2', type=int, default=5)
    parser.add_argument('--B3', help='the target of regression task 3', type=int, default=3)
    #parser.add_argument('--B1_x', help='the mean of input distribution for task 1', type=int, default=1)
    #parser.add_argument('--B2_x', help='the mean of input distribution for task 2', type=int, default=1)
    #parser.add_argument('--B3_x', help='the mean of input distribution for task 3', type=int, default=1)
    parser.add_argument('--inner_lr', help='lr for inner update', type=float, default=0.01)
    parser.add_argument('--outer_lr', help='lr for outer update', type=float, default=0.01)
    parser.add_argument('--first_order', help='use first order approximation', action='store_true')
    parser.add_argument('--plot', help='whether plot figure', action='store_true')
    parser.add_argument('--dir', help='log directory', type=str,default='./data/regress_toy/')
    parser.add_argument('--number', help='number for storing data', type=int, default=5)
    parser.add_argument('--batch_size', help='batch size', type=int, default=8)

    args = parser.parse_args()
    log_dir=args.dir+str(args.num_layers)+'_'+str(2)+'_'+str(args.inner_lr)+'_'+str(
        args.outer_lr)+'_'+str(args.B1)+'_'+str(args.B2)+'_'+str(args.B3)+'_'+str(args.batch_size)+'clip1/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    main(args,log_dir)