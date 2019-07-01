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

class network(nn.Module):
    def __init__(self,num_layers,num_hidden,num_out):
        super(network,self).__init__()
        self.add_module('layer0', nn.Linear(1, num_hidden))
        for i in range(1,num_layers-1):
            self.add_module('layer{0}'.format(i),nn.Linear(num_hidden,num_hidden))
        self.add_module('layer{0}'.format(num_layers-1), nn.Linear(num_hidden, num_out))
        self.apply(weight_init)
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