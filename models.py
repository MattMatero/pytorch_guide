import torch
import torch.nn as nn
from torch.autograd import Variable


class my_gru(nn.Module):

    def __init__(self, input_size, hidden_size, input_drop, output_drop, bidirect=True, num_layers=1):
        super(my_gru, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirect)
        
        self.bidirect = bidirect
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.idrop = nn.Dropout(p=input_drop)
        self.odrop = nn.Dropout(p=output_drop)
        self.nlayers = num_layers

        if self.bidirect:
            self.linear = nn.Linear(self.hidden_size*2,1)
        else:
            self.linear = nn.Linear(self.hidden_size,1)
        
    def forward(self, inputs, hidden):
        out, self.hidden = self.gru(self.idrop(inputs),hidden)
        decoded = self.linear(self.odrop(out)) 
        return decoded, self.hidden
    
    def init_hidden(self):
        weight = next(self.parameters())
        if self.bidirect:
            return weight.new_zeros(2*self.nlayers, 1, self.hidden_size) # 2 for bi-directional

        return weight.new_zeros(self.nlayers, 1, self.hidden_size)

class my_lstm(nn.Module):

    def __init__(self, input_size, hidden_size, input_drop, output_drop, bidirect=True, num_layers=1):
        super(my_lstm, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirect)
        
        self.bidirect = bidirect
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.idrop = nn.Dropout(p=input_drop)
        self.odrop = nn.Dropout(p=output_drop)
        self.nlayers = num_layers
        #hidden0 = self.init_hidden()
        #self.hidden0 = nn.Parameter(hidden0)
        self.hidden = self.init_hidden()

        if self.bidirect:
            self.linear = nn.Linear(self.hidden_size*2,1)
        else:
            self.linear = nn.Linear(self.hidden_size,1)

    def forward(self, inputs, hidden):
        if hidden is None:
            hidden = self.hidden

        out, self.hidden = self.lstm(self.idrop(inputs),hidden)
        decoded = self.linear(self.odrop(out)) # manually apply dropout on outputs
        return decoded, self.hidden
    
    def init_hidden(self):
        weight = next(self.parameters())
        if self.bidirect:
            return (weight.new_zeros(2*self.nlayers, 1, self.hidden_size).cuda(),weight.new_zeros(2*self.nlayers, 1, self.hidden_size).cuda()) # 2 for bi-directional

        return (weight.new_zeros(self.nlayers, 1, self.hidden_size).cuda(),weight.new_zeros(self.nlayers, 1, self.hidden_size).cuda())
