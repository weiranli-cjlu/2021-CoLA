import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super().__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == "prelu" else act
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_ft))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out = out + self.bias
        return self.act(out)


class AvgReadout(nn.Module):
    def forward(self, seq):
        return torch.mean(seq, dim=1)


class MaxReadout(nn.Module):
    def forward(self, seq):
        return torch.max(seq, dim=1).values


class MinReadout(nn.Module):
    def forward(self, seq):
        return torch.min(seq, dim=1).values


class WSReadout(nn.Module):
    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        # 原实现固定 repeat(..., 64)，当 embedding_dim != 64 时会出错。
        sim = sim.repeat(1, 1, seq.size(-1))
        return torch.sum(seq * sim, dim=1)


class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super().__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        self.negsamp_round = negsamp_round
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.f_k.weight)
        if self.f_k.bias is not None:
            nn.init.zeros_(self.f_k.bias)

    def forward(self, c, h_pl):
        scores = [self.f_k(h_pl, c)]
        c_mi = c
        # torch.roll 比 torch.cat((x[-2:-1], x[:-1])) 少一次 Python 级拼接。
        for _ in range(self.negsamp_round):
            c_mi = torch.roll(c_mi, shifts=1, dims=0)
            scores.append(self.f_k(h_pl, c_mi))
        return torch.cat(scores, dim=0)


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout):
        super().__init__()
        self.read_mode = readout
        self.gcn = GCN(n_in, n_h, activation)
        if readout == "max":
            self.read = MaxReadout()
        elif readout == "min":
            self.read = MinReadout()
        elif readout == "avg":
            self.read = AvgReadout()
        elif readout == "weighted_sum":
            self.read = WSReadout()
        else:
            raise ValueError(f"Unsupported readout: {readout}")
        self.disc = Discriminator(n_h, negsamp_round)

    def forward(self, seq1, adj, sparse=False):
        h_1 = self.gcn(seq1, adj, sparse)
        if self.read_mode != "weighted_sum":
            c = self.read(h_1[:, :-1, :])
            h_mv = h_1[:, -1, :]
        else:
            h_mv = h_1[:, -1, :]
            c = self.read(h_1[:, :-1, :], h_1[:, -2:-1, :])
        return self.disc(c, h_mv)
