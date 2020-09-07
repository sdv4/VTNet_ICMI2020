import pickle
import os
import random
import shutil

import numpy as np

import torch
from torch import nn
from torch import utils
from torch import autograd
from torchvision import datasets

from utils import pickle_loader
from utils import plot_metrics

from train import train

MANUAL_SEED = 1

np.random.seed(MANUAL_SEED)
random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


class ConfusionRNN(nn.Module):
    """
        Basic/Vanilla RNN.
    """
    def __init__(self, input_size, hidden_size, output_size, batch_size, num_layers=1):
        super(ConfusionRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(1)

    def get_architecture(self):
        """ Helper method that returns model type """
        return 'rnn'
        
    def forward(self, inputs, hidden):
        # change input shape to (max_seq_size,batch_size,input_features):
        inputs = inputs.permute(1, 0, 2)
        output, hidden = self.rnn(inputs, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self, batch_size):
        """ Initializes the hidden state with zero tensors.
        """
        return autograd.Variable(torch.zeros(self.num_layers, batch_size,
                                             self.hidden_size))

class ConfusionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, num_layers=1, dropout=0.0):

        super(ConfusionLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout)

        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(1)

    def get_architecture(self):
        """ Helper method that returns model type """
        return 'lstm'

    def forward(self, inputs, hidden):
        # change input shape to (max_seq_size,batch_size,input_features):
        inputs = inputs.permute(1, 0, 2)
        output, hidden = self.lstm(inputs, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self, batch_size):
        """ Initializes the hidden state and cell state with zero tensors.
        """
        return (autograd.Variable(torch.zeros(self.num_layers, batch_size,
                                              self.hidden_size)),
                autograd.Variable(torch.zeros(self.num_layers, batch_size,
                                              self.hidden_size)))

class ConfusionGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, num_layers=1):
        super(ConfusionGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(1)

    def get_architecture(self):
        """ Helper method that returns model type """
        return 'gru'
    
    def forward(self, inputs, hidden):
        # change input shape to (max_seq_size,batch_size,input_features):
        inputs = inputs.permute(1, 0, 2)
        output, hidden = self.gru(inputs, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self, batch_size):
        """ Initializes the hidden state with zero tensors.
        """
        return autograd.Variable(torch.zeros(self.num_layers, batch_size,
                                             self.hidden_size))
    
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        self.softmax = nn.LogSoftmax(1)
        
    def get_architecture(self):
        """ Helper method that returns model type """
        return 'FFNN'
    
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        out = self.softmax(y_pred)
        return out