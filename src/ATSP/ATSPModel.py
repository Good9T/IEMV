import torch
import torch.nn as nn
import torch.nn.functional as F

from ATSPModel_Lib import AddAndInstanceNormalization, FeedForward, MixedScoreAttention, get_encoding, reshape_by_heads

class ATSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.encoder = ATSP_Encoder(**self.model_params)
        self.decoder = ATSP_Decoder(**self.model_params)
        self.encoded_row = None
        self.encoded_col = None

    def pre_forward(self, reset_state):
        problems = reset_state.problems

        B = problems.size(0)
        N = problems.size(1)
        E = int(self.model_params['embedding_dim'])

        row_emb = torch.zeros(B, N, E)
        col_emb = torch.zeros(B, N, E)

        seed_num = int(self.model_params['one_hot_seed_cnt'])
        rand = torch.rand(B, seed_num)
        batch_rand_perm = rand.argsort(dim=1)

        rand_idx = batch_rand_perm[:, :N]
        batch_idx = torch.arange(B)[:, None].expand(B, N)
        node_idx = torch.arange(N)[None, :].expand(B, N)
        col_emb[batch_idx, node_idx, rand_idx] = 1

        self.encoded_row, self.encoded_col = self.encoder(row_emb, col_emb, problems)
        self.decoder.set_kv(self.encoded_col)

    def forward(self, state):
        B = state.BATCH_IDX.size(0)
        P = state.BATCH_IDX.size(1)

        if state.selected_count == 0:
            selected = torch.arange(P)[None, :].expand(B, P)
            prob = torch.ones(B, P)
            encoded_first_row = get_encoding(self.encoded_row, selected)
            self.decoder.set_q1(encoded_first_row)

        else:
            encoded_current_row = get_encoding(self.encoded_row, state.current_node)
            all_probs = self.decoder(encoded_current_row, mask=state.state_mask)

            if self.training:
                while True:
                    with torch.no_grad():
                        selected = all_probs.reshape(B * P, -1).multinomial(1).squeeze(dim=1).reshape(B, P)
                    prob = all_probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(B, P)
                    if (prob != 0).all():
                        break
            else:
                selected = all_probs.argmax(dim=2)
                prob = None

        return selected, prob

class ATSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        encoder_layer_num = int(model_params["encoder_layer_num"])
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, row_emb, col_emb, problem):
        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, problem)

        return row_emb, col_emb

class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.row_encoding_block = EncodingBlock(**model_params)
        self.col_encoding_block = EncodingBlock(**model_params)

    def forward(self, row_emb, col_emb, problem):
        row_emb_out = self.row_encoding_block(row_emb, col_emb, problem)
        col_emb_out = self.col_encoding_block(col_emb, row_emb, problem.transpose(1, 2))
        return row_emb_out, col_emb_out

class EncodingBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.E = int(model_params["embedding_dim"])
        self.H = int(model_params["head_num"])
        self.D = int(model_params["qkv_dim"])

        self.Wq = nn.Linear(self.E, self.H * self.D, bias=False)
        self.Wk = nn.Linear(self.E, self.H * self.D, bias=False)
        self.Wv = nn.Linear(self.E, self.H * self.D, bias=False)

        self.mixed_score_attention = MixedScoreAttention(**model_params)
        self.multi_head_combine = nn.Linear(self.H * self.D, self.E)

        self.normalization1 = AddAndInstanceNormalization(**model_params)
        self.feedforward = FeedForward(**model_params)
        self.normalization2 = AddAndInstanceNormalization(**model_params)

    def forward(self, row, col, problem):
        q = reshape_by_heads(self.Wq(row), head_num=self.H)
        k = reshape_by_heads(self.Wk(col), head_num=self.H)
        v = reshape_by_heads(self.Wv(col), head_num=self.H)

        out_concat = self.mixed_score_attention(q, k, v, problem)
        mixed_score_attention = self.multi_head_combine(out_concat)

        out1 = self.normalization1(row, mixed_score_attention)
        out2 = self.feedforward(out1)
        out3 = self.normalization2(out1, out2)

        return out3

class ATSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.E = int(model_params["embedding_dim"])
        self.H = int(model_params["head_num"])
        self.D = int(model_params["qkv_dim"])
        self.SCALE_D = int(model_params["sqrt_qkv_dim"])
        self.SCALE_E = int(model_params["sqrt_embedding_dim"])
        self.clip = model_params["logit_clipping"]

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
        self.k = reshape_by_heads(self.Wk(encoded_col), head_num=self.H)
        self.v = reshape_by_heads(self.Wv(encoded_col), head_num=self.H)

        self.single_head_key = encoded_col.transpose(1, 2)

    def set_q1(self, first_row):
        self.q1 = reshape_by_heads(self.Wq1(first_row), head_num=self.H)

    def _decoder_attention(self, q, k, v, mask=None):
        B = q.size(0)
        n = q.size(2)
        N = k.size(2)

        score = torch.matmul(q, k.transpose(2, 3))
        score = score / self.SCALE_D

        if mask is not None:
            score = score + mask[:, None, :, :].expand(B, self.H, n, N)

        weights = torch.softmax(score, dim=3)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2)
        out = out.reshape(B, n, self.H * self.D)

        return out

    def forward(self, q0, mask):
        q0 = reshape_by_heads(self.Wq0(q0), head_num=self.H)
        q = self.q1 + q0

        out = self._decoder_attention(q, self.k, self.v, mask=mask)
        score = self.multi_head_combine(out)

        score = torch.matmul(score, self.single_head_key)
        score = score / self.SCALE_E
        score = self.clip * torch.tanh(score)
        score = score + mask

        probs = F.softmax(score, dim=2)

        return probs