import torch
import torch.nn as nn
import torch.nn.functional as F

class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        E = int(model_params['embedding_dim'])
        self.norm = nn.InstanceNorm1d(E, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        added = input1 + input2
        added = added.transpose(1, 2)
        added = self.norm(added)
        added = added.transpose(1, 2)
        return added

class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        E = int(model_params['embedding_dim'])
        h_dim = int(model_params['ff_hidden_dim'])

        self.W1 = nn.Linear(E, h_dim)
        self.W2 = nn.Linear(h_dim, E)

    def forward(self, x):
        out = F.relu(self.W1(x))
        out = self.W2(out)
        return out

class MixedScoreAttention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.H = int(model_params['head_num'])
        self.D = int(model_params['qkv_dim'])
        self.SCALE_D  = int(model_params['sqrt_qkv_dim'])
        self.ms_hidden = int(model_params['ms_hidden_dim'])

        mix1_init = model_params['ms_layer1_init']
        mix2_init = model_params['ms_layer2_init']

        self.w1 = nn.Parameter(torch.empty(self.H, 2, self.ms_hidden).uniform_(-mix1_init, mix1_init))
        self.b1 = nn.Parameter(torch.empty(self.H, self.ms_hidden).uniform_(-mix1_init, mix1_init))

        self.w2 = nn.Parameter(torch.empty(self.H, self.ms_hidden, 1).uniform_(-mix2_init, mix2_init))
        self.b2 = nn.Parameter(torch.empty(self.H, 1).uniform_(-mix2_init, mix2_init))

    def forward(self, q, k, v, cost_mat):
        B = q.size(0)
        H = q.size(1)
        N = q.size(2)
        D = q.size(3)

        col_n = k.size(2)

        dot = torch.matmul(q, k.transpose(2, 3))
        dot_score = dot / self.SCALE_D

        cost_expand = cost_mat[:, None, :, :]
        cost_score = cost_expand.expand(B, H, N, col_n)

        cat = torch.stack([dot_score, cost_score], dim=4)
        cat = cat.transpose(1, 2)

        x = torch.matmul(cat, self.w1)
        x = x + self.b1[None, None, :, None]
        x = F.relu(x)

        x = torch.matmul(x, self.w2)
        x = x + self.b2[None, None, :, None]
        x = x.transpose(1, 2)
        x = x.squeeze(4)

        weights = torch.softmax(x, dim=3)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2)
        out = out.reshape(B, N, H * D)

        return out

def reshape_by_heads(qkv, head_num):
    H = int(head_num)
    B = qkv.size(0)
    N = qkv.size(1)

    out = qkv.reshape(B, N, H, -1)
    out = out.transpose(1, 2)
    return out

def get_encoding(encoded_nodes, idx):
    B = idx.size(0)
    P = idx.size(1)
    E = encoded_nodes.size(2)

    gather_idx = idx[:, :, None].expand(B, P, E)
    out = encoded_nodes.gather(dim=1, index=gather_idx)
    return out