import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_net import Base_net
from collections import OrderedDict
from layers import *
class Fast_net(Base_net):
    def __init__(self,loss_fn,num_updates, step_size, batch_size, meta_batch_size):
        super(Fast_net, self).__init__(loss_fn)
        self.num_updates = num_updates
        self.step_size = step_size
        self.batch_size = batch_size
        self.meta_batch_size = meta_batch_size

    def forward_pass(self, in_, target, weights=None):
        ''' Run data through net, return loss and output '''
        # Run the batch through the net, compute loss
        out = self.forward_ori(in_, weights)
        loss = self.loss_fn(out, target)
        return loss, out

    def forward(self, sample_x,sample_y,sample_x_meta,sample_y_meta,num_updates):
        self.num_updates = num_updates
        fast_weights = OrderedDict((name, param) for (name, param) in self.named_parameters())
        for i in range(self.num_updates):
            #print('inner step', i)
            if i == 0:
                loss, _ = self.forward_pass(sample_x, sample_y)
                grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
            else:
                loss, _ = self.forward_pass(sample_x, sample_y, fast_weights)
                grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            #print(loss)
            fast_weights = OrderedDict(
                (name, param - self.step_size * grad) for ((name, param), grad) in zip(fast_weights.items(), grads))
        ##### Test net after training, should be better than random ####


        # Compute the meta gradient and return it
        loss, _ = self.forward_pass(sample_x_meta, sample_y_meta, fast_weights)
        loss = loss / self.meta_batch_size  # normalize loss
        grads = torch.autograd.grad(loss, self.parameters())
        meta_grads = {name: g for ((name, _), g) in zip(self.named_parameters(), grads)}

        return  meta_grads,loss