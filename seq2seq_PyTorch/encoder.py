import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import MAX_LENGTH

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.GRU = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded= self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.GRU(output, hidden)
        return output, hidden

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

if __name__ == '__main__':
    encoder = EncoderRNN()