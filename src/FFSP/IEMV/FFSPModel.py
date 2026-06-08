import torch
import torch.nn as nn
import torch.nn.functional as F
from FFSPModel_Lib import AddAndInstanceNormalization, FeedForward, MixedScoreAttention

class FFSPModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.S = int(model_params['stage_cnt'])
        # One stage model for each stage
        self.stage_models = nn.ModuleList([
            OneStageModel(stage_idx, **model_params)
            for stage_idx in range(self.S)
        ])

    def pre_forward(self, reset_state):
        for stage_idx in range(self.S):
            problems = reset_state.problems_list[stage_idx]
            model = self.stage_models[stage_idx]
            model.pre_forward(problems)

    def soft_reset(self):
        pass

    def forward(self, state):
        B = state.BATCH_IDX.size(0)
        P = state.BATCH_IDX.size(1)

        action_stack = torch.empty(size=(B, P, self.S), dtype=torch.long)
        prob_stack = torch.empty(size=(B, P, self.S))

        for stage_idx in range(self.S):
            model = self.stage_models[stage_idx]
            action, prob = model(state)

            action_stack[:, :, stage_idx] = action
            prob_stack[:, :, stage_idx] = prob

        gathering_index = state.stage_idx[:, :, None]
        action = action_stack.gather(dim=2, index=gathering_index).squeeze(dim=2)
        prob = prob_stack.gather(dim=2, index=gathering_index).squeeze(dim=2)

        return action, prob

class OneStageModel(nn.Module):
    def __init__(self, stage_idx, **model_params):
        super().__init__()
        self.model_params = model_params
        self.machine_cnt_list = model_params['machine_cnt_list']
        self.machine_cnt = self.machine_cnt_list[stage_idx]  # machines in current stage
        self.embedding_dim = int(model_params['embedding_dim'])  # E
        self.seed_cnt = int(model_params['one_hot_seed_cnt'])

        self.encoder = FFSP_Encoder(**model_params)
        self.decoder = FFSP_Decoder(**model_params)

        self.encoded_row = None  # (B, J, E) encoded jobs
        self.encoded_col = None  # (B, M, E) encoded machines

    def pre_forward(self, problems):
        B = problems.size(0)
        J = problems.size(1)
        M = problems.size(2)

        row_emb = torch.zeros(size=(B, J, self.embedding_dim))
        col_emb = torch.zeros(size=(B, M, self.embedding_dim))

        rand = torch.rand(size=(B, self.seed_cnt))
        batch_rand_perm = rand.argsort(dim=1)
        rand_idx = batch_rand_perm[:, :M]

        b_idx = torch.arange(B)[:, None].expand(B, M)
        m_idx = torch.arange(M)[None, :].expand(B, M)

        col_emb[b_idx, m_idx, rand_idx] = 1
        self.encoded_row, self.encoded_col = self.encoder(row_emb, col_emb, problems)
        self.decoder.set_kv(self.encoded_row)

    def forward(self, state):
        B = state.BATCH_IDX.size(0)
        P = state.BATCH_IDX.size(1)
        # Get current machine encoding
        machine_emb = self._get_encoding(self.encoded_col, state.stage_machine_idx)  # (B,P,E)
        # Decode job probabilities
        job_probs = self.decoder(machine_emb, ninf_mask=state.job_ninf_mask)  # (B,P,J+1)
        # shape: (batch, pomo, job)
        if self.training or self.model_params['eval_type'] == 'softmax':
            while True:
                job_selected = job_probs.reshape(B * P, -1).multinomial(1)
                job_selected = job_selected.squeeze(1).reshape(B, P)

                job_prob = job_probs[state.BATCH_IDX, state.POMO_IDX, job_selected].reshape(B, P)
                job_prob[state.finished] = 1.0

                if (job_prob != 0).all():
                    break
        else:
            job_selected = job_probs.argmax(dim=2)
            job_prob = torch.zeros((B, P))

        return job_selected, job_prob

    def _get_encoding(self, encoded_nodes, node_index_to_pick):
        B = node_index_to_pick.size(0)
        P = node_index_to_pick.size(1)
        gathering_index = node_index_to_pick[:, :, None].expand(B, P, self.embedding_dim)
        picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index) # (B,P,E)
        return picked_nodes

class FFSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        layer_num = model_params['encoder_layer_num']
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(layer_num)])

    def forward(self, row_emb, col_emb, cost_mat):
        # row_emb: (B,J,E)   col_emb: (B,M,E)   cost_mat: (B,J,M)
        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, cost_mat)

        return row_emb, col_emb


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.row_encoding_block = EncodingBlock(**model_params)
        self.col_encoding_block = EncodingBlock(**model_params)

    def forward(self, row_emb, col_emb, cost_mat):
        # row_emb: (B,J,E)   col_emb: (B,M,E)   cost_mat: (B,J,M)
        row_emb_out = self.row_encoding_block(row_emb, col_emb, cost_mat)
        col_emb_out = self.col_encoding_block(col_emb, row_emb, cost_mat.transpose(1, 2))

        return row_emb_out, col_emb_out


class EncodingBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.E = int(model_params['embedding_dim'])
        self.H = int(model_params['head_num'])
        self.D = int(model_params['qkv_dim'])

        self.Wq = nn.Linear(self.E, self.H * self.D, bias=False)
        self.Wk = nn.Linear(self.E, self.H * self.D, bias=False)
        self.Wv = nn.Linear(self.E, self.H * self.D, bias=False)

        self.mixed_score_MHA = MixedScoreAttention(**model_params)
        self.multi_head_combine = nn.Linear(self.H * self.D, self.E)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, row_emb, col_emb, cost_mat):
        # row_emb: (B,J,E)   col_emb: (B,M,E)   cost_mat: (B,J,M)

        q = reshape_by_heads(self.Wq(row_emb), head_num=self.H)  # (B,H,J,D)
        k = reshape_by_heads(self.Wk(col_emb), head_num=self.H)  # (B,H,M,D)
        v = reshape_by_heads(self.Wv(col_emb), head_num=self.H)  # (B,H,M,D)

        out_concat = self.mixed_score_MHA(q, k, v, cost_mat)  # (B,J,H*D)
        multi_head_out = self.multi_head_combine(out_concat)  # (B,J,E)

        out1 = self.add_n_normalization_1(row_emb, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3


class FFSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        E = int(model_params['embedding_dim'])
        H = int(model_params['head_num'])
        D = int(model_params['qkv_dim'])

        self.embedding_dim = E
        self.head_num = H
        self.qkv_dim = D
        self.sqrt_qkv_dim = int(model_params['sqrt_qkv_dim'])
        self.logit_clipping = model_params['logit_clipping']

        self.encoded_NO_JOB = nn.Parameter(torch.rand(1, 1, E))  # (1,1,E)

        self.Wq_1 = nn.Linear(E, H * D, bias=False)
        self.Wq_2 = nn.Linear(E, H * D, bias=False)
        self.Wq_3 = nn.Linear(E, H * D, bias=False)
        self.Wk = nn.Linear(E, H * D, bias=False)
        self.Wv = nn.Linear(E, H * D, bias=False)

        self.multi_head_combine = nn.Linear(H * D, E)

        self.k = None
        self.v = None
        self.single_head_key = None

    def set_kv(self, encoded_job):
        # encoded_job: (B,J,E)
        B = encoded_job.size(0)
        J = encoded_job.size(1)
        E = encoded_job.size(2)

        encoded_no_job = self.encoded_NO_JOB.expand(B, 1, E)  # (B,1,E)
        encoded_jobs_plus_1 = torch.cat([encoded_job, encoded_no_job], dim=1)  # (B,J+1,E)

        self.k = reshape_by_heads(self.Wk(encoded_jobs_plus_1), head_num=self.head_num)  # (B,H,J+1,D)
        self.v = reshape_by_heads(self.Wv(encoded_jobs_plus_1), head_num=self.head_num)  # (B,H,J+1,D)

        self.single_head_key = encoded_jobs_plus_1.transpose(1, 2)  # (B,E,J+1)

    def forward(self, encoded_machine, ninf_mask):
        # encoded_machine: (B,P,E)   ninf_mask: (B,P,J+1)

        q = reshape_by_heads(self.Wq_3(encoded_machine), head_num=self.head_num)  # (B,H,P,D)
        out_concat = self._multi_head_attention_for_decoder(q, self.k, self.v, rank3_ninf_mask=ninf_mask)  # (B,P,H*D)

        mh_atten_out = self.multi_head_combine(out_concat)  # (B,P,E)

        score = torch.matmul(mh_atten_out, self.single_head_key) / self.sqrt_qkv_dim
        score_clipped = self.logit_clipping * torch.tanh(score)
        score_masked = score_clipped + ninf_mask
        probs = F.softmax(score_masked, dim=2)  # (B,P,J+1)

        return probs

    def _multi_head_attention_for_decoder(self, q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
        B = q.size(0)
        H = q.size(1)
        N = q.size(2)
        D = q.size(3)
        J_plus = k.size(2)  # job count + 1

        score = torch.matmul(q, k.transpose(2, 3)) / self.sqrt_qkv_dim  # (B,H,N,J+)

        if rank2_ninf_mask is not None:
            score += rank2_ninf_mask[:, None, None, :].expand(B, H, N, J_plus)
        if rank3_ninf_mask is not None:
            score += rank3_ninf_mask[:, None, :, :].expand(B, H, N, J_plus)

        weights = F.softmax(score, dim=3)
        out = torch.matmul(weights, v)  # (B,H,N,D)

        out_transposed = out.transpose(1, 2)  # (B,N,H,D)
        out_concat = out_transposed.reshape(B, N, H * D)  # (B,N,H*D)

        return out_concat





def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE
    batch_size = qkv.size()[0]
    n = qkv.size()[1]
    q_reshaped = qkv.reshape(batch_size, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)
    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)
    return q_transposed