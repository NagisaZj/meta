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