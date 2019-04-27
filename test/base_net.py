import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *

class Base_net(nn.Module):
    def __init__(self,loss_fn):
        super(Base_net, self).__init__()
        self.dense1 = nn.Linear(1,40)
        self.dense2 = nn.Linear(40,1)
        self.dense1.weight.data.normal_(0, 0.1)
        self.dense2.weight.data.normal_(0, 0.1)
        self.loss_fn = loss_fn
        return

    def forward_ori(self,x,weights=None):
        if weights==None:
            tem = self.dense1(x)
            tem = F.relu(tem)
            tem = self.dense2(tem)
        else:
            '''self.dense1.weight.data = weights['dense1.weight']
            self.dense2.weight.data = weights['dense2.weight']
            self.dense1.bias.data = weights['dense1.bias']
            self.dense2.bias.data = weights['dense2.bias']'''
            tem = linear(x,weights['dense1.weight'],weights['dense1.bias'])
            tem = F.relu(tem)
            tem = linear(tem,weights['dense2.weight'],weights['dense2.bias'])
        return tem

    def copy_weights(self, net):
        ''' Set this module's weights to be the same as those of 'net' '''
        # TODO: breaks if nets are not identical
        # TODO: won't copy buffers, e.g. for batch norm
        for m_from, m_to in zip(net.modules(), self.modules()):
            if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()