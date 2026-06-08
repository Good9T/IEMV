import torch
import torch.nn as nn
import torch.nn.functional as F


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        E = int(model_params['embedding_dim'])  # Embedding Dim
        self.norm = nn.InstanceNorm1d(E, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        added = input1 + input2
        transposed = added.transpose(1, 2)
        normalized = self.norm(transposed)
        back_trans = normalized.transpose(1, 2)
        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        E = int(model_params['embedding_dim'])  # Embedding Dim
        ff_hidden_dim = int(model_params['ff_hidden_dim'])

        self.W1 = nn.Linear(E, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, E)

    def forward(self, input1):
        out = F.relu(self.W1(input1))
        out = self.W2(out)
        return out


class MixedScoreAttention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.head_num = int(model_params['head_num'])
        self.D = int(model_params['qkv_dim'])            # QKV Dim
        self.sqrt_qkv_dim = int(model_params['sqrt_qkv_dim'])
        self.ms_hidden_dim = int(model_params['ms_hidden_dim'])
        self.mix1_init = model_params['ms_layer1_init']
        self.mix2_init = model_params['ms_layer2_init']

        mix1_weight = torch.distributions.Uniform(low=-self.mix1_init, high=self.mix1_init).sample((self.head_num, int(2), self.ms_hidden_dim))
        mix1_bias = torch.distributions.Uniform(low=-self.mix1_init, high=self.mix1_init).sample((self.head_num, self.ms_hidden_dim))
        self.mix1_weight = nn.Parameter(mix1_weight)
        self.mix1_bias = nn.Parameter(mix1_bias)

        mix2_weight = torch.distributions.Uniform(low=-self.mix2_init, high=self.mix2_init).sample((self.head_num, self.ms_hidden_dim, int(1)))
        mix2_bias = torch.distributions.Uniform(low=-self.mix2_init, high=self.mix2_init).sample((self.head_num, int(1)))
        self.mix2_weight = nn.Parameter(mix2_weight)
        self.mix2_bias = nn.Parameter(mix2_bias)

    def forward(self, q, k, v, cost_mat):
        B = q.size(0)        # Batch
        H = q.size(1)        # Head
        N = q.size(2)        # Node
        D = q.size(3)        # QKV Dim
        N_k = k.size(2)      # Node

        dot_product = torch.matmul(q, k.transpose(2, 3))
        dot_product_score = dot_product / self.sqrt_qkv_dim

        cost_mat_score = cost_mat[:, None, :, :].expand(B, H, N, N_k)
        stack_score = torch.stack((dot_product_score, cost_mat_score), dim=4)

        stack_score_transposed = stack_score.transpose(1, 2)
        ms1 = torch.matmul(stack_score_transposed, self.mix1_weight)
        ms1 = ms1 + self.mix1_bias[None, None, :, None]
        ms1_activated = F.relu(ms1)

        ms2 = torch.matmul(ms1_activated, self.mix2_weight)
        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]
        mixed_score = ms2.transpose(1, 2)
        mixed_scores = mixed_score.squeeze(4)

        weights = torch.softmax(mixed_scores, dim=3)
        out = torch.matmul(weights, v)
        out_transposed = out.transpose(1, 2)
        out_concat = out_transposed.reshape(B, N, H * D)

        return out_concat


def reshape_by_heads(qkv, head_num):
    B = qkv.size(0)        # Batch
    N = qkv.size(1)        # Node
    H = int(head_num)      # Head

    out = qkv.reshape(B, N, H, -1)
    out = out.transpose(1, 2)

    return out


def _get_encoding(encoded_nodes, node_index_to_pick):
    B = node_index_to_pick.size(0)    # Batch
    P = node_index_to_pick.size(1)    # Pomo
    E = encoded_nodes.size(2)         # Embedding Dim

    gathering_index = node_index_to_pick[:, :, None].expand(B, P, E)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)

    return picked_nodes