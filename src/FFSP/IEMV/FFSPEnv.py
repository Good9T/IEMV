import torch
from dataclasses import dataclass
import itertools
from yaml import full_load
from FFSPProblem import get_random_problems

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.patches as patches

@dataclass
class Reset_State:
    problems_list: list  # shape: [ (B, J, M0), (B, J, M1), ... ]

@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor    # (B, P)
    POMO_IDX: torch.Tensor     # (B, P)
    stage_idx: torch.Tensor = None    # (B, P)
    stage_machine_idx: torch.Tensor = None
    finished: torch.Tensor = None     # (B, P)
    job_ninf_mask: torch.Tensor = None    # (B, P, J+1)
    step_cnt: int = 0

class FFSPEnv:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.stage_cnt = env_params['stage_cnt']
        self.machine_cnt_list = env_params['machine_cnt_list']
        self.total_machine_cnt = sum(self.machine_cnt_list)
        self.job_cnt = env_params['job_cnt']
        self.process_time_params = env_params['process_time_params']
        self.pomo_size = env_params['pomo_size']
        self.sm_indexer = _Stage_N_Machine_Index_Converter(self)

        # View
        self.problems = None
        self.trans_problems = None
        self.problems_list = None  # list[(B, J, M)]
        self.trans_problems_list = None

        # Const
        self.batch_size = None
        self.BATCH_IDX = None  # (B, P)
        self.POMO_IDX = None  # (B, P)

        self.job_durations = None  # (B, J+1, total_M)
        # last job means NO_JOB ==> duration = 0

        # Dynamic
        self.time_idx = None
        self.sub_time_idx = None  # 0 ~ total_M-1
        self.machine_idx = None  # (B, P)

        self.schedule = None  # (B, P, M, J+1)
        self.machine_wait_step = None  # (B, P, M)
        self.job_location = None  # (B, P, J+1)
        self.job_wait_step = None  # (B, P, J+1)
        self.finished = None  # (B, P)

        self.step_state = None

    def load_problems(self, batch_size, aug_enable=False, aug_factor=1):
        self.batch_size = batch_size

        self.problems, self.trans_problems = get_random_problems(
            batch_size,
            self.stage_cnt,
            self.machine_cnt_list,
            self.job_cnt,
            self.process_time_params
        )

        self.problems_list = [p.float() for p in self.problems]
        self.trans_problems_list = [p.float() for p in self.trans_problems]

        if aug_enable:
            self.problems_list = [p.repeat(aug_factor, 1, 1) for p in self.problems_list]
            self.trans_problems_list = [p.repeat(aug_factor, 1, 1) for p in self.trans_problems_list]
            self.batch_size *= aug_factor

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.job_durations = torch.empty(
            (self.batch_size, self.job_cnt + 1, self.total_machine_cnt),
            dtype=torch.long
        )
        self.job_durations[:, :self.job_cnt, :] = torch.cat(self.problems_list, dim=2)
        self.job_durations[:, self.job_cnt, :] = 0

    def load_equivalent_view(self):
        self.problems_list = self.trans_problems_list
        self.job_durations[:, :self.job_cnt, :] = torch.cat(self.trans_problems_list, dim=2)

    def reset(self):
        self.time_idx = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.long)
        self.sub_time_idx = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.long)
        self.machine_idx = self.sm_indexer.get_machine_index(self.POMO_IDX, self.sub_time_idx) # (B, P)

        self.schedule = torch.full(size=(self.batch_size, self.pomo_size, self.total_machine_cnt, self.job_cnt+1),
                                   dtype=torch.long, fill_value=-999999)
        self.machine_wait_step = torch.zeros(size=(self.batch_size, self.pomo_size, self.total_machine_cnt), dtype=torch.long)
        self.job_location = torch.zeros(size=(self.batch_size, self.pomo_size, self.job_cnt+1), dtype=torch.long)
        self.job_wait_step = torch.zeros(size=(self.batch_size, self.pomo_size, self.job_cnt+1), dtype=torch.long)
        self.finished = torch.full(size=(self.batch_size, self.pomo_size), dtype=torch.bool, fill_value=False)
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)

        reward = None
        done = None
        return Reset_State(self.problems_list), reward, done

    def pre_step(self):
        self._update_step_state()
        self.step_state.step_cnt = 0
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, job_idx):
        self.schedule[self.BATCH_IDX, self.POMO_IDX, self.machine_idx, job_idx] = self.time_idx

        job_length = self.job_durations[self.BATCH_IDX, job_idx, self.machine_idx]
        self.machine_wait_step[self.BATCH_IDX, self.POMO_IDX, self.machine_idx] = job_length
        self.job_location[self.BATCH_IDX, self.POMO_IDX, job_idx] += 1
        self.job_wait_step[self.BATCH_IDX, self.POMO_IDX, job_idx] = job_length

        self.finished = (self.job_location[:, :, :self.job_cnt] == self.stage_cnt).all(dim=2)
        done = self.finished.all()

        if not done:
            self._move_to_next_machine()
            self._update_step_state()

        reward = -self._get_makespan() if done else None
        return self.step_state, reward, done

    def _move_to_next_machine(self):
        b_idx = torch.flatten(self.BATCH_IDX)  # (B*P,) batch index
        p_idx = torch.flatten(self.POMO_IDX)  # (B*P,) pomo index
        ready = torch.flatten(self.finished)  # (B*P,) done flag

        b_idx = b_idx[~ready]
        p_idx = p_idx[~ready]

        while not ready.all():
            new_sub_t = self.sub_time_idx[b_idx, p_idx] + 1  # new sub‑time step
            step_inc = new_sub_t == self.total_machine_cnt  # whether to increment global time
            self.time_idx[b_idx, p_idx] += step_inc.long()
            new_sub_t[step_inc] = 0
            self.sub_time_idx[b_idx, p_idx] = new_sub_t
            new_machine = self.sm_indexer.get_machine_index(p_idx, new_sub_t)  # new machine index
            self.machine_idx[b_idx, p_idx] = new_machine

            m_wait = self.machine_wait_step[b_idx, p_idx, :]  # machine remaining waiting steps
            m_wait[step_inc, :] -= 1
            m_wait[m_wait < 0] = 0
            self.machine_wait_step[b_idx, p_idx, :] = m_wait

            j_wait = self.job_wait_step[b_idx, p_idx, :]  # job remaining waiting steps
            j_wait[step_inc, :] -= 1
            j_wait[j_wait < 0] = 0
            self.job_wait_step[b_idx, p_idx, :] = j_wait

            machine_ready = self.machine_wait_step[b_idx, p_idx, new_machine] == 0
            new_stage = self.sm_indexer.get_stage_index(new_sub_t)  # (N,) current stage index

            job_loc_ok = self.job_location[b_idx, p_idx, :self.job_cnt] == new_stage[:, None]  # (N,J)
            job_wait_ok = self.job_wait_step[b_idx, p_idx, :self.job_cnt] == 0
            job_ready = (job_loc_ok & job_wait_ok).any(1)  # (N,)
            ready = machine_ready & job_ready

            b_idx = b_idx[~ready]
            p_idx = p_idx[~ready]


    def _update_step_state(self):
        self.step_state.step_cnt += 1
        self.step_state.stage_idx = self.sm_indexer.get_stage_index(self.sub_time_idx)
        self.step_state.stage_machine_idx = self.sm_indexer.get_stage_machine_index(self.POMO_IDX, self.sub_time_idx)

        job_loc = self.job_location[:, :, :self.job_cnt]  # (B,P,J) job location
        job_wait_t = self.job_wait_step[:, :, :self.job_cnt]  # (B,P,J) job remaining time

        job_in_stage = job_loc == self.step_state.stage_idx[:, :, None]  # (B,P,J)
        job_not_waiting = (job_wait_t == 0)
        job_available = job_in_stage & job_not_waiting  # (B,P,J) available jobs

        job_prev = (job_loc < self.step_state.stage_idx[:, :, None]).any(2)  # (B,P)
        job_wait_stage = (job_in_stage & (job_wait_t > 0)).any(2)  # (B,P)
        wait_allowed = job_prev | job_wait_stage | self.finished  # (B,P)

        self.step_state.job_ninf_mask = torch.full((self.batch_size, self.pomo_size, self.job_cnt + 1), float('-inf'))
        job_enable = torch.cat([job_available, wait_allowed.unsqueeze(2)], dim=2)
        self.step_state.job_ninf_mask[job_enable] = 0

        self.step_state.finished = self.finished

    def _get_makespan(self):
        job_dur_perm = self.job_durations.permute(0, 2, 1)  # (B, M, J+1) permuted job duration
        end_sched = self.schedule + job_dur_perm[:, None, :, :]  # (B, P, M, J+1) end time of jobs
        end_time_max, _ = end_sched[:, :, :, :self.job_cnt].max(dim=3)  # (B, P, M)
        end_time_max, _ = end_time_max.max(dim=2)  # (B, P) final makespan
        return end_time_max


class _Stage_N_Machine_Index_Converter:
    def __init__(self, env):
        self.machine_cnt_list = env.machine_cnt_list
        self.stage_cnt = env.stage_cnt
        self.pomo_size = env.pomo_size

        sub_tables = []
        machine_tables = []
        current_start = 0

        for m_cnt in self.machine_cnt_list:
            sub_index = torch.tensor(list(itertools.permutations(range(m_cnt))))
            machine_index = sub_index + current_start
            sub_tables.append(sub_index)
            machine_tables.append(machine_index)
            current_start += m_cnt

        self.machine_SUBindex_table = torch.cat(sub_tables, dim=1)
        self.machine_table = torch.cat(machine_tables, dim=1)

        stage_table = []
        for stage_id, m_cnt in enumerate(self.machine_cnt_list):
            stage_table += [stage_id] * m_cnt
        self.stage_table = torch.tensor(stage_table, dtype=torch.long)

    def get_stage_index(self, sub_time_idx):
        return self.stage_table[sub_time_idx]
    def get_machine_index(self, POMO_idx, sub_time_idx):
        return self.machine_table[POMO_idx, sub_time_idx]
    def get_stage_machine_index(self, POMO_IDX, sub_time_idx):
        return self.machine_SUBindex_table[POMO_IDX, sub_time_idx]

