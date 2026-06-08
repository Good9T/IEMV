import torch
import torch.nn as nn
import torch.nn.functional as F


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input1, input2: (B,P,E)
        added = input1 + input2

        transposed = added.transpose(1, 2)  # (B,E,P)
        normalized = self.norm(transposed)
        back_trans = normalized.transpose(1, 2)  # (B,P,E)

        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        out = F.relu(self.W1(input1))
        out = self.W2(out)

        return out


class MixedScoreAttention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.head_num = int(model_params['head_num'])
        self.ms_hidden_dim = int(model_params['ms_hidden_dim'])
        self.mix1_init = model_params['ms_layer1_init']
        self.mix2_init = model_params['ms_layer2_init']
        self.qkv_dim = int(model_params['qkv_dim'])
        self.sqrt_qkv_dim = int(model_params['sqrt_qkv_dim'])

        mix1_weight = torch.distributions.Uniform(low=-self.mix1_init, high=self.mix1_init).sample((self.head_num, 2, self.ms_hidden_dim))
        mix1_bias = torch.distributions.Uniform(low=-self.mix1_init, high=self.mix1_init).sample((self.head_num, self.ms_hidden_dim))
        self.mix1_weight = nn.Parameter(mix1_weight)  # (H,2,ms_hidden)
        self.mix1_bias = nn.Parameter(mix1_bias)      # (H,ms_hidden)

        mix2_weight = torch.distributions.Uniform(low=-self.mix2_init, high=self.mix2_init).sample((self.head_num, self.ms_hidden_dim, 1))
        mix2_bias = torch.distributions.Uniform(low=-self.mix2_init, high=self.mix2_init).sample((self.head_num, 1))
        self.mix2_weight = nn.Parameter(mix2_weight)  # (H,ms_hidden,1)
        self.mix2_bias = nn.Parameter(mix2_bias)      # (H,1)

    def forward(self, q, k, v, cost_mat):
        # q: (B,H,J,D)    k,v: (B,H,M,D)    cost_mat: (B,J,M)
        B = q.size(0)
        H = q.size(1)
        J = q.size(2)
        M = k.size(2)
        D = q.size(3)

        dot_product = torch.matmul(q, k.transpose(2, 3))
        dot_product_score = dot_product / self.sqrt_qkv_dim

        cost_mat_score = cost_mat[:, None, :, :].expand(B, H, J, M)  # (B,H,J,M)
        stack_score = torch.stack((dot_product_score, cost_mat_score), dim=4)  # (B,H,J,M,2)

        stack_score_transposed = stack_score.transpose(1, 2)  # (B,J,H,M,2)
        ms1 = torch.matmul(stack_score_transposed, self.mix1_weight)
        ms1 = ms1 + self.mix1_bias[None, None, :, None]
        ms1_activated = F.relu(ms1)

        ms2 = torch.matmul(ms1_activated, self.mix2_weight)
        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]
        mixed_score = ms2.transpose(1, 2)  # (B,H,J,M,1)

        mixed_scores = mixed_score.squeeze(4)
        weights = torch.softmax(mixed_scores, dim=3)  # (B,H,J,M)

        out = torch.matmul(weights, v)  # (B,H,J,D)
        out_transposed = out.transpose(1, 2)  # (B,J,H,D)
        out_concat = out_transposed.reshape(B, J, H * D)

        return out_concat