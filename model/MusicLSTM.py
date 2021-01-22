import torch
from torch import nn
import numpy as np


class MusicLSTM(nn.Module):
    def __init__(self, n_tokens, seq_length=None, d_model=64,
            n_heads=4, depth=2, d_feedforward=512, dropout=0.1,
            positional_encoding=False, relative_pos=True):
        super().__init__()
        self.number_layer = 2
        self.vocab_size = 413
        self.lstm = nn.LSTM(self.vocab_size, self.vocab_size, self.number_layer)

        self.n_tokens = n_tokens

    def forward(self, x):
        """
        :param x: bxseq_length index of note event
        :return: bx413xseq_length probability of note event
        """
        x = x.permute(1, 0)
        sequence_length, batch_size = x.shape
        one_hot = torch.nn.functional.one_hot(x, self.vocab_size)
        one_hot = one_hot.float()
        h0 = torch.randn(self.number_layer, batch_size, self.vocab_size).cuda()
        c0 = torch.randn(self.number_layer, batch_size, self.vocab_size).cuda()
        output, _ = self.lstm(one_hot, (h0, c0))
        output = output.permute(1, 2, 0)
        output = torch.softmax(output, dim=1)
        return output