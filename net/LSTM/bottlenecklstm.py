from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
class BottleneckLSTMCell(nn.Module):
    """ Creates a LSTM layer cell
    Arguments:
        input_channels : variable used to contain value of number of channels in input
        hidden_channels : variable used to contain value of number of channels in the hidden state of LSTM cell
    """
    def __init__(self, input_channels, hidden_channels):
        super(BottleneckLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_features = 4
        self.W = nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels, kernel_size=3, groups=self.input_channels, stride=1, padding=1)
        self.Wy  = nn.Conv2d(int(self.input_channels+self.hidden_channels), self.hidden_channels, kernel_size=1)
        self.Wi  = nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, 1, 1, groups=self.hidden_channels, bias=False)  
        self.Wbi = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
        self.Wbf = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
        self.Wbc = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
        self.Wbo = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
        self.ac = nn.ReLU6()

        # self.Wci = None
        # self.Wcf = None
        # self.Wco = None
#         logging.info("Initializing weights of lstm")
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Returns:
            initialized weights of the model
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            
    def forward(self, x, h, c): #implemented as mentioned in paper here the only difference is  Wbi, Wbf, Wbc & Wbo are commuted all together in paper
        """
        Arguments:
            x : input tensor
            h : hidden state tensor
            c : cell state tensor
        Returns:
            output tensor after LSTM cell 
        """
        x = self.W(x)
        y = torch.cat((x, h),1) #concatenate input and hidden layers
        i = self.Wy(y)  #reduce to hidden layer size
        b = self.Wi(i)    #depth wise 3*3
        ci = torch.sigmoid(self.Wbi(b))
        cf = torch.sigmoid(self.Wbf(b))
        cc = cf * c + ci * F.tanh(self.Wbc(b))
        co = torch.sigmoid(self.Wbo(b))
        ch = co * F.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        """
        Arguments:
            batch_size : an int variable having value of batch size while training
            hidden : an int variable having value of number of channels in hidden state
            shape : an array containing shape of the hidden and cell state 
        Returns:
            cell state and hidden state
        """
        # if self.Wci is None:
        #     self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        #     self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        #     self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        # else:
        #     assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
        #     assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda()
                )

class BottleneckLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, height, width):
        """ Creates Bottleneck LSTM layer
        Arguments:
            input_channels : variable having value of number of channels of input to this layer
            hidden_channels : variable having value of number of channels of hidden state of this layer
            height : an int variable having value of height of the input
            width : an int variable having value of width of the input
            batch_size : an int variable having value of batch_size of the input
        Returns:
            Output tensor of LSTM layer
        """
        super(BottleneckLSTM, self).__init__()
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.cell = BottleneckLSTMCell(self.input_channels, self.hidden_channels)
        
#         self.hidden_state = h
#         self.cell_state = c
#         print(h.size(),c.size()

    def forward(self, input):
#         new_h, new_c = self.cell(input, self.hidden_state, self.cell_state)
#         self.hidden_state = new_h
#         self.cell_state = new_c
        batch_size,seq_len,_,height, width = input.size()
        h, c = self.cell.init_hidden(batch_size, hidden=self.hidden_channels, shape=(height, width))
#         h, c = self.cell(input[:,0], self.hidden_state, self.cell_state)
        output_inner = []
        for t in range(seq_len):
            h, c = self.cell(input[:, t, :, :, :], h, c)
            output_inner.append(h)
        layer_output = torch.stack(output_inner, dim=1)
        return layer_output