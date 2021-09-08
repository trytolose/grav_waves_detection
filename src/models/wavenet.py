import torch
import torch.nn as nn


class Wave_Block_true(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates):
        super(Wave_Block_true, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.dil_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.dil_convs.append(
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=dilation_rate,
                    dilation=dilation_rate,
                )
            )
            self.filter_convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1))
            self.gate_convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

        self.end_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.convs[0](x)
        #         res = x
        skip = 0
        for i in range(self.num_rates):

            res = x
            x = self.dil_convs[i](x)
            x = torch.mul(
                torch.tanh(self.filter_convs[i](x)),
                torch.sigmoid(self.gate_convs[i](x)),
            )
            x = self.convs[i + 1](x)
            skip = skip + x
            # x += res
            x = x + res

        x = self.end_block(skip)
        return x


class Andrewnet_v3_true(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels

        self.first_conv = nn.Sequential(nn.Conv1d(in_channels, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU())
        self.waveblock_1 = nn.Sequential(Wave_Block_true(64, 16, 12), nn.BatchNorm1d(16))
        self.waveblock_2 = nn.Sequential(Wave_Block_true(16, 32, 8), nn.BatchNorm1d(32))
        self.waveblock_3 = nn.Sequential(Wave_Block_true(32, 64, 4), nn.BatchNorm1d(64))
        self.waveblock_4 = nn.Sequential(Wave_Block_true(64, 128, 1), nn.BatchNorm1d(128))
        self.waveblock_5 = nn.Sequential(nn.Conv1d(128, 128, 7, padding=3), nn.BatchNorm1d(128), nn.ReLU())

        self.dropout = nn.Dropout(p=0.2)
        # self.attn = Attention(4096, 128)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.waveblock_1(x)
        x = self.waveblock_2(x)
        x = self.waveblock_3(x)
        x = self.waveblock_4(x)
        x = self.waveblock_5(x)
        # x = self.dropout(x)
        x = self.pool(x)
        # x = self.attn(x.transpose(2,1))
        x = self.fc(x.view(-1, 128))
        return x

class Andrewnet_v4(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels

        self.first_conv = nn.Sequential(nn.Conv1d(in_channels, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU())
        self.waveblock_1 = nn.Sequential(Wave_Block_true(16, 16, 8), nn.BatchNorm1d(16))
        self.waveblock_2 = nn.Sequential(Wave_Block_true(16, 32, 4), nn.BatchNorm1d(32))
        self.waveblock_3 = nn.Sequential(Wave_Block_true(32, 64, 2), nn.BatchNorm1d(64))
        self.waveblock_4 = nn.Sequential(Wave_Block_true(64, 128, 1), nn.BatchNorm1d(128))
        # self.waveblock_5 = nn.Sequential(nn.Conv1d(128, 128, 7, padding=3), nn.BatchNorm1d(128), nn.ReLU())
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.waveblock_1(x)
        x = self.waveblock_2(x)
        x = self.waveblock_3(x)
        x = self.waveblock_4(x)
        # x = self.waveblock_5(x)
        # x = self.dropout(x)
        x = self.pool(x)
        # x = self.attn(x.transpose(2,1))
        x = self.fc(x.view(-1, 128))
        return x



class Andrewnet_v5(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels

        self.first_conv = nn.Sequential(nn.Conv1d(in_channels, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU())
        self.waveblock_1 = nn.Sequential(Wave_Block_true(16, 16, 8), nn.BatchNorm1d(16))
        self.waveblock_2 = nn.Sequential(Wave_Block_true(16, 32, 4), nn.BatchNorm1d(32))
        self.waveblock_3 = nn.Sequential(Wave_Block_true(32, 64, 2), nn.BatchNorm1d(64))
        self.waveblock_4 = nn.Sequential(Wave_Block_true(64, 128, 1), nn.BatchNorm1d(128))
        self.pool = nn.MaxPool1d(8)
        self.fc1 = nn.Sequential(nn.Linear(128, 1), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(512, 1), nn.ReLU())

    def forward(self, x):
        bs = x.shape[0]
        x = self.first_conv(x)
        x = self.waveblock_1(x)
        x = self.waveblock_2(x)
        x = self.waveblock_3(x)
        x = self.waveblock_4(x)

        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = x.view(bs, -1)
        x = self.fc2(x)
        
        return x






class Attention(nn.Module):
    def __init__(self, step_dim, features_dim):
        
        super(Attention, self).__init__()
        self.features_dim = features_dim
        self.step_dim = step_dim

        self.weight = torch.nn.Parameter(data=torch.Tensor(features_dim, 1), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.weight)

        self.b = torch.nn.Parameter(data=torch.Tensor(step_dim, 1), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.b)
        
    def forward(self, x):
        w = self.weight
        b = torch.reshape(self.b, (-1,))
        x_new = torch.reshape(x,(-1, x.shape[-1]))
        eij = torch.matmul(x_new,w).reshape(-1, self.step_dim)
        eij = eij+b
        eij = torch.tanh(eij)
        eij = torch.exp(eij)
        
        a = torch.div(eij,(torch.sum(eij, 1,keepdim=True)+1e-07)).type(torch.float)    # + epsilon
        a = torch.reshape(a,(-1,self.step_dim, 1))
        a = x*a
        
        return torch.sum(a, 1)

# class RNN(nn.Module):
#     def __init__(self):
#         super(RNN, self).__init__()

#         self.rnn0 = nn.LSTM(input_size=INPUT_SIZE, hidden_size=128, batch_first=True, bidirectional=True)
#         self.rnn1 = nn.LSTM(input_size=256,hidden_size=64, batch_first=True, bidirectional=True)
#         self.attn = Attention(160, 128)
#         self.out0 = nn.Linear(128, 64)
#         self.out1 = nn.Linear(64, 1)
#         self.relu0 = nn.ReLU()

#     def forward(self, x):

#         r_out, (h_n, h_c) = self.rnn0(x, None)  
#         r_out, _  = self.rnn1(r_out, None)
#         out = self.attn(r_out)
    
      
#         out = torch.reshape(out, (-1,128))
#         out = self.out0(out)
#         out = self.relu0(out)
#         out = self.out1(out)
        
#         return torch.sigmoid(out)