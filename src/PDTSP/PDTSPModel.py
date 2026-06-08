import torch
import torch.nn as nn
import torch.nn.functional as F

from PDTSPModel_Lib import AddAndInstanceNormalization, FeedForward, MixedScoreAttention, _get_encoding, reshape_by_heads


class PDTSPModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.encoder = PDTSP_Encoder(**model_params)
        self.decoder = PDTSP_Decoder(**model_params)
        self.E = int(model_params['embedding_dim'])  # Embedding Dim

        self.encoded_row = None
        self.encoded_col = None

    def pre_forward(self, reset_state):
        problems = reset_state.problems

        B = problems.size(0)  # Batch
        N = problems.size(1)  # Node

        row_emb = torch.zeros((B, N, self.E), device=problems.device)
        col_emb = torch.zeros((B, N, self.E), device=problems.device)

        customer_size = int((N - 1) // 2)

        rand = torch.rand(B, customer_size, device=problems.device)
        batch_rand_perm = rand.argsort(dim=1) + 1
        rand_idx = torch.zeros(B, N, dtype=torch.long, device=problems.device)

        rand_idx[:, 0] = 0
        rand_idx[:, 1: 1 + customer_size] = batch_rand_perm
        rand_idx[:, 1 + customer_size:] = batch_rand_perm + customer_size

        batch_idx = torch.arange(B, device=problems.device)[:, None].expand(B, N)
        node_idx = torch.arange(N, device=problems.device)[None, :].expand(B, N)
        col_emb[batch_idx, node_idx, rand_idx] = 1

        self.encoded_row, self.encoded_col = self.encoder(row_emb, col_emb, problems)
        self.decoder.set_kv(self.encoded_col)

    def forward(self, state):
        B = state.BATCH_IDX.size(0)  # Batch
        P = state.BATCH_IDX.size(1)  # Pomo

        if state.selected_count == 0:
            selected = torch.arange(P)[None, :].expand(B, P)
            prob = torch.ones(B, P)

            encoded_first_row = _get_encoding(self.encoded_row, selected)
            self.decoder.set_q1(encoded_first_row)

        else:
            encoded_current_row = _get_encoding(self.encoded_row, state.current_node)
            all_probs = self.decoder(encoded_current_row, mask=state.state_mask)

            if self.training:
                while True:
                    with torch.no_grad():
                        selected = all_probs.flatten(0, 1).multinomial(1).view(B, P)
                    prob = all_probs[state.BATCH_IDX, state.POMO_IDX, selected]
                    if (prob != 0).all():
                        break
            else:
                selected = all_probs.argmax(dim=2)
                prob = None

        return selected, prob


class PDTSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        layer_num = int(model_params["encoder_layer_num"])
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(layer_num)])

    def forward(self, row_emb, col_emb, problem):
        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, problem)
        return row_emb, col_emb


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.row_block = EncodingBlock(**model_params)
        self.col_block = EncodingBlock(**model_params)

    def forward(self, row_emb, col_emb, problem):
        row_out = self.row_block(row_emb, col_emb, problem)
        col_out = self.col_block(col_emb, row_emb, problem.transpose(1, 2))
        return row_out, col_out


class EncodingBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.E = int(model_params["embedding_dim"])         # Embedding Dim
        self.H = int(model_params["head_num"])             # Head
        self.D = int(model_params["qkv_dim"])              # QKV Dim

        self.Wq = nn.Linear(self.E, self.H * self.D, bias=False)
        self.Wk = nn.Linear(self.E, self.H * self.D, bias=False)
        self.Wv = nn.Linear(self.E, self.H * self.D, bias=False)

        self.attention = MixedScoreAttention(**model_params)
        self.multi_head_combine = nn.Linear(self.H * self.D, self.E)

        self.norm1 = AddAndInstanceNormalization(**model_params)
        self.ff = FeedForward(**model_params)
        self.norm2 = AddAndInstanceNormalization(**model_params)

    def forward(self, row, col, cost_matrix):
        q = reshape_by_heads(self.Wq(row), self.H)
        k = reshape_by_heads(self.Wk(col), self.H)
        v = reshape_by_heads(self.Wv(col), self.H)

        out = self.attention(q, k, v, cost_matrix)
        out = self.multi_head_combine(out)

        out = self.norm1(row, out)
        out = self.norm2(out, self.ff(out))
        return out


class PDTSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.E = int(model_params["embedding_dim"])         # Embedding Dim
        self.H = int(model_params["head_num"])             # Head
        self.D = int(model_params["qkv_dim"])              # QKV Dim

        self.Wq0 = nn.Linear(self.E, self.H * self.D, bias=False)
        self.Wq1 = nn.Linear(self.E, self.H * self.D, bias=False)
        self.Wk = nn.Linear(self.E, self.H * self.D, bias=False)
        self.Wv = nn.Linear(self.E, self.H * self.D, bias=False)
        self.multi_head_combine = nn.Linear(self.H * self.D, self.E)

        self.k = None
        self.v = None
        self.single_head_key = None
        self.q1 = None

    def set_kv(self, encoded_col):
        self.k = reshape_by_heads(self.Wk(encoded_col), self.H)
        self.v = reshape_by_heads(self.Wv(encoded_col), self.H)
        self.single_head_key = encoded_col.transpose(1, 2)

    def set_q1(self, first_row):
        self.q1 = reshape_by_heads(self.Wq1(first_row), self.H)

    def _attention(self, q, k, v, mask):
        B = q.size(0)  # Batch
        H = q.size(1)  # Head
        N = q.size(2)  # Node
        node_num = k.size(2)

        score = torch.matmul(q, k.transpose(2, 3)) / int(self.model_params["sqrt_qkv_dim"])

        if mask is not None:
            score = score + mask[:, None, :, :].expand(B, H, N, node_num)

        weights = score.softmax(dim=-1)
        out = torch.matmul(weights, v).transpose(1, 2).flatten(2)
        return out

    def forward(self, q0, mask):
        q0 = reshape_by_heads(self.Wq0(q0), self.H)
        q = self.q1 + q0

        out = self._attention(q, self.k, self.v, mask)
        out = self.multi_head_combine(out)

        score = torch.matmul(out, self.single_head_key)
        score = score / int(self.model_params["sqrt_embedding_dim"])
        score = self.model_params["logit_clipping"] * torch.tanh(score)
        score = score + mask

        return F.softmax(score, dim=-1)