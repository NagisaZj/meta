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
    def __init__(self,num_b,allow_bias):
        super(network,self).__init__()
        for i in range(num_b):
            self.add_module('layer{0}'.format(i),nn.Linear(1,2,bias=allow_bias))
        if allow_bias:
            self.apply(weight_init_bias)
        else:
            self.apply(weight_init)
        self.num_b = num_b
        self.allow_bias = allow_bias

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = input
        for i in range(self.num_b):
            output = F.linear(output,
                              weight=params['layer{0}.weight'.format(i)],
                              bias=torch.zeros(1,) if not self.allow_bias else params['layer{0}.bias'.format(i)])
        output = torch.sum(output,1)
        output = torch.unsqueeze(output,1)
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

def learn(learner,args,task_target,log_dir):
    if args.num_b==1:
        loss_rem = []
        b1_rem = []
        b2_rem = []
        b1_after_rem = []
        b2_after_rem = []
        b1_gradients=[]
        b2_gradients=[]
        for i in range(args.num_tasks):
            b1_after_rem.append([])
            b2_after_rem.append([])
        for update in range(args.num_updates):
            input = torch.FloatTensor(torch.ones(1, 1))
            lossfun = nn.MSELoss()
            losses = []
            b1_rem.append(learner.layer0.weight.data.numpy()[0, 0])
            b2_rem.append(learner.layer0.weight.data.numpy()[1, 0])
            for i in range(args.num_tasks):

                target = torch.FloatTensor(torch.ones(1, 1) * task_target[i])
                loss = lossfun(learner.forward(input), target)
                params = learner.update_params(loss, step_size=args.inner_lr, first_order=args.first_order)
                b1_after_rem[i].append(params['layer0.weight'].data.numpy()[0,0])
                b2_after_rem[i].append(params['layer0.weight'].data.numpy()[1, 0])
                print('b1 after%i' % i, params['layer0.weight'].data.numpy()[0, 0])
                print('b2 after%i' % i, params['layer0.weight'].data.numpy()[1, 0])
                #print(params['layer0.weight'].data.numpy()[0,0])
                losses.append(lossfun(learner.forward(input, params), target))

            total_loss = torch.mean(torch.stack(losses, dim=0))
            loss_rem.append(total_loss.data.numpy())
            grads = torch.autograd.grad(total_loss, learner.parameters())
            grads = parameters_to_vector(grads)
            b1_gradients.append(grads[0].data.numpy() * args.outer_lr * -1)
            b2_gradients.append(grads[1].data.numpy() * args.outer_lr * -1)
            old_params = parameters_to_vector(learner.parameters())
            vector_to_parameters(old_params - args.outer_lr * grads, learner.parameters())


            print('loss',total_loss.data.numpy())
            print('b1 before',learner.layer0.weight.data.numpy()[0,0])
            print('b2 before', learner.layer0.weight.data.numpy()[1, 0])
        np.save(log_dir+'loss.npy',loss_rem)
        np.save(log_dir+'b1.npy',b1_rem)
        np.save(log_dir + 'b1after1.npy', b1_after_rem[0])
        np.save(log_dir + 'b1after2.npy', b1_after_rem[1])
        np.save(log_dir + 'b2.npy', b2_rem)
        np.save(log_dir + 'b2after1.npy', b2_after_rem[0])
        np.save(log_dir + 'b2after2.npy', b2_after_rem[1])
        plt.figure()
        plt.plot(loss_rem)
        plt.title('mean loss(after adaption)')
        plt.figure()
        plt.plot(b1_rem,label='b1(before update)')
        plt.plot(b1_after_rem[0], label='b1(after update on task 1 (3))')
        plt.plot(b1_after_rem[1], label='b1(after update on task 2 (5))')
        plt.legend()
        plt.figure()
        plt.plot(b2_rem, label='b2(before update)')
        plt.plot(b2_after_rem[0], label='b2(after update on task 1 (3))')
        plt.plot(b2_after_rem[1], label='b2(after update on task 2 (5))')
        plt.legend()
        xs = np.arange(0, 5, 0.5)
        y1s = 3 - xs
        y2s = 5 - xs
        '''for i in range(50):
            plt.figure()
            plt.scatter(xs, y1s)
            plt.scatter(xs, y2s)
            plt.scatter(b1_rem[i], b2_rem[i])
            plt.arrow(b1_rem[i], b2_rem[i], b1_gradients[i], b2_gradients[i])
            plt.scatter(b1_after_rem[0][i], b2_after_rem[0][i])
            plt.scatter(b1_after_rem[1][i], b2_after_rem[1][i])
            plt.savefig(log_dir + 'figures/%i.png' % i)

            plt.close()'''
        plt.show()


    elif args.num_b==2:
        loss_rem = []
        b1_rem = []
        b2_rem = []
        b1_after_rem = []
        b2_after_rem = []
        for i in range(args.num_tasks):
            b1_after_rem.append([])
            b2_after_rem.append([])
        for update in range(args.num_updates):
            input = torch.FloatTensor(torch.ones(1, 1))
            lossfun = nn.MSELoss()
            losses = []
            for i in range(args.num_tasks):
                target = torch.FloatTensor(torch.ones(1, 1) * task_target[i])
                loss = lossfun(learner.forward(input), target)
                params = learner.update_params(loss, step_size=args.inner_lr, first_order=args.first_order)
                b1_after_rem[i].append(params['layer0.weight'].data.numpy()[0, 0])
                b2_after_rem[i].append(params['layer1.weight'].data.numpy()[0, 0])
                print('b1 after%i'%i,params['layer0.weight'].data.numpy()[0,0])
                print('b2 after%i' % i, params['layer1.weight'].data.numpy()[0, 0])
                losses.append(lossfun(learner.forward(input, params), target))

            total_loss = torch.mean(torch.stack(losses, dim=0))
            loss_rem.append(total_loss.data.numpy())
            grads = torch.autograd.grad(total_loss, learner.parameters())
            grads = parameters_to_vector(grads)
            old_params = parameters_to_vector(learner.parameters())
            vector_to_parameters(old_params - args.outer_lr * grads, learner.parameters())
            b1_rem.append(learner.layer0.weight.data.numpy()[0, 0])
            b2_rem.append(learner.layer1.weight.data.numpy()[0, 0])

            print('loss',total_loss.data.numpy())
            print('b1 before',learner.layer0.weight.data.numpy()[0, 0])
            print('b2 before',learner.layer1.weight.data.numpy()[0, 0])
            np.save(log_dir + 'loss.npy', loss_rem)
            np.save(log_dir + 'b1.npy', b1_rem)
            np.save(log_dir + 'b2.npy', b2_rem)
            np.save(log_dir + 'b1after1.npy', b1_after_rem[0])
            np.save(log_dir + 'b1after2.npy', b1_after_rem[1])
            np.save(log_dir + 'b2after1.npy', b2_after_rem[0])
            np.save(log_dir + 'b2after2.npy', b2_after_rem[1])
            plt.figure()
            plt.plot(loss_rem)
            plt.title('mean loss(after adaption)')
            plt.figure()
            plt.plot(b1_rem, label='b1(before update)')
            plt.plot(b1_after_rem[0], label='b1(after update on task 1 (3))')
            plt.plot(b1_after_rem[1], label='b1(after update on task 2 (5))')
            plt.legend()
            plt.figure()
            plt.plot(b2_rem, label='b2(before update)')
            plt.plot(b2_after_rem[0], label='b2(after update on task 1 (3))')
            plt.plot(b2_after_rem[1], label='b2(after update on task 2 (5))')
            plt.legend()
            plt.show()
            

    return



def main(args,log_dir):
    learner = network(args.num_b,args.allow_bias)
    task_target = [args.B1, args.B2, args.B3]
    learn(learner,args,task_target,log_dir)
        #save(args)

    return






if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_updates', help='number of updates', type=int, default=200)
    parser.add_argument('--num_b', help='number of network parameters b', type=int, default=1)
    parser.add_argument('--B1',help='the target of regression task 1',type=int,default=3)
    parser.add_argument('--B2', help='the target of regression task 2', type=int, default=5)
    parser.add_argument('--B3', help='the target of regression task 3', type=int, default=7)
    parser.add_argument('--num_tasks', help='number of regression tasks', type=int, default=2)
    parser.add_argument('--inner_lr', help='lr for inner update', type=float, default=0.1)
    parser.add_argument('--outer_lr', help='lr for outer update', type=float, default=0.1)
    parser.add_argument('--first_order', help='use first order approximation', action='store_true')
    parser.add_argument('--plot', help='whether plot figure', action='store_true')
    parser.add_argument('--allow_bias', help='allowing bias', action='store_true')
    parser.add_argument('--dir', help='log directory', type=str,default='./data/linear_toy/')

    args = parser.parse_args()
    log_dir=args.dir+str(args.num_b)+'_'+str(args.num_tasks)+'_'+str(args.inner_lr)+'_'+str(
        args.outer_lr)+'_'+str(args.B1)+'_'+str(args.B2)+'_'+str(args.B3)+'add/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    main(args,log_dir)