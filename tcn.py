import math
import torch
import torch.nn as nn

from locked_dropout import LockedDropout

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCN(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernal_size, stride, dilation, dropout=0.2):
        super(TCN, self).__init__()
        padding = (kernal_size - 1) * dilation
        # self.lockdrop = LockedDropout()
        # padding = int((kernal_size - 1) / 2)
        # '''
        self.conv = nn.Conv1d(n_inputs, n_outputs, kernal_size,
                stride=stride, padding=padding, dilation=dilation)
        # self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernal_size,
        #         stride=stride, padding=padding, dilation=dilation)
        self.chomp = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        # self.dropout = nn.Dropout(0.2)
        # self.relu2 = nn.ReLU()
        # self.conv_net = nn.Sequential(self.conv, self.relu1)
        # self.conv_net = nn.Sequential(self.conv, self.relu1,
        #         self.conv2, self.relu2)
        self.conv_net = nn.Sequential(self.conv, self.chomp, self.relu1)
        # self.conv_net = nn.Sequential(self.conv, self.chomp,
        #         self.relu1, self.dropout)
        # '''
        self.linear = nn.Linear(n_outputs, 1)
        # self.relu_linear = nn.ReLU()
        self.linear_net = nn.Sequential(self.linear)
        # self.linear_net = nn.Sequential(self.linear, self.relu_linear)
        self.hardtanh1 = nn.Hardtanh()
        self.hardtanh2 = nn.Hardtanh()

        self.n_outputs_rsqrt = 1 / math.sqrt(n_outputs)

        # self.softmax = nn.Softmax(dim=1)
        self.temp = 10.0

    def ht(self, ht_func, x):
        return 0.5 * (ht_func(x * self.temp) + 1)
        # return 0.5 * (ht_func(x * self.temp - 1 / self.temp) + 1)

    def forward(self, x, seq_len_data):
        seq_len = x.size(0)
        batch_size = x.size(1)

        # x = self.lockdrop(x, 0.4)

        ones = torch.ones(seq_len, seq_len).cuda()
        shifter_down = ones.tril(-1) - ones.tril(-2)
        shifter_up = ones.triu(1) - ones.triu(2)
        # '''
        x = x.transpose(0, 1).transpose(1, 2)
        x_conv = self.conv_net(x)
        x_conv = x_conv.transpose(1, 2).transpose(0, 1)
        # '''
        # x_conv = x
        x_output = self.linear_net(x_conv).squeeze(2)
        # x_output *= self.n_outputs_rsqrt
        # print(x_output)
        
        x_shift_down = torch.mm(shifter_down, x_output)
        x_shift_up = torch.mm(shifter_up, x_output)
        
        mask = ones.tril().unsqueeze(2)
        mask2 = ones.tril(1).unsqueeze(2)
        mask_shift = ones.tril(1).unsqueeze(2)
        mask_shift2 = ones.tril(2).unsqueeze(2)
        
        x_row = x_shift_down.unsqueeze(0)
        x_column = x_output.unsqueeze(1)
        # x_column = x_shift_up.unsqueeze(1)
        x_square1 = x_row - x_column
        # square_1 = (x_square1 < 0).float()
        square_1 = self.ht(self.hardtanh1, x_square1) * (1 - mask_shift)
        # square_1 = self.ht(self.hardtanh1, x_square1 * (1 - mask_shift2))

        # square_2 = (x_output - x_shift_down < 0).float().unsqueeze(0)
        square_2 = self.ht(self.hardtanh2, x_shift_down - x_output).unsqueeze(0)
        x_span_split_index = square_1 * square_2

        all_ones = torch.ones_like(x_span_split_index)
        span = all_ones - x_span_split_index
        
        span = (mask + (1 - mask) * span).cumprod(dim=1) * (1 - mask)
        # span = (mask2 + (1 - mask2) * span).cumprod(dim=1) * (1 - mask)

        # Soft Softmax
        # x_att = self.softmax(x_output.unsqueeze(0).unsqueeze(3))
        # attention = span.unsqueeze(3) * x_att
        
        '''
        # Hard Softmax
        span_scores = span.unsqueeze(3) * (x_output.unsqueeze(0).unsqueeze(3))
        mask_zero = (span_scores != 0).float()
        span_scores += mask_zero.log()
        span_scores = torch.cat([span_scores, torch.ones(seq_len, 1, batch_size, 1).cuda() * -10], 1)
        attention_raw = self.softmax(span_scores)
        attention, _ = attention_raw.split([seq_len, 1], 1)
        '''
        
        # Sigmoid
        # span_scores = span.unsqueeze(3) * x_output.sigmoid().unsqueeze(0).unsqueeze(3)
        # attention = span_scores / (span_scores.sum(1, keepdim=True) + 1e-4)
        
        # Linear Normalize
        x_att = x_output - x_output.min() + 10
        span_scores = span.unsqueeze(3) * x_att.unsqueeze(0).unsqueeze(3)
        # span_scores *= (1 - ones.triu(10)).unsqueeze(0).unsqueeze(3)
        if seq_len == seq_len_data:
            seq_len_data -= 1
        span_scores = span_scores[:seq_len_data]
        
        # len_mask = ones.triu(10).unsqueeze(2)
        # reg_len = (span * len_mask).pow(2).mean()
        reg_len = x_output.pow(2).mean()

        # span_scores *= (1 - len_mask[:seq_len_data].unsqueeze(3))
        
        attention = span_scores / (span_scores.sum(1, keepdim=True))
        # attention = span_scores / (span_scores.sum(1, keepdim=True) + 1e-4)
        
        # attention = attention[:seq_len_data]
        # attention = span_scores / (span_scores.sum(1, keepdim=True) + 1e-4)
        return attention, seq_len_data, reg_len, # x_output


    def forward_raw(self, x):
        seq_len = x.size(0)
        x = x.transpose(0, 1).transpose(1, 2)
        x_conv = self.conv_net(x)
        x_conv = x_conv.transpose(1, 2).transpose(0, 1)
        x_output = self.linear_net(x_conv)
        x_row = x_output.unsqueeze(0)
        x_column = x_output.unsqueeze(1)
        x_square = self.ht(x_column - x_row)
        mask = torch.ones(seq_len, seq_len).cuda().tril()
        mask_right = torch.ones(seq_len, seq_len).cuda().triu(diagonal=10)
        mask = mask.unsqueeze(2).unsqueeze(3)
        mask_right = mask_right.unsqueeze(2).unsqueeze(3)
        # x_square *= mask_right
        x_square = mask + (1 - mask) * x_square
        x_square = x_square.cumprod(dim=1) * (1 - mask)
        return x_square
