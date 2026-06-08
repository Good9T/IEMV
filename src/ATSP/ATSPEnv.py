from dataclasses import dataclass
import torch
from ATSPProblem import get_random_problems

@dataclass
class Reset_State:
    problems: torch.Tensor    # (B, N, N)

@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor    # (B, P)
    POMO_IDX: torch.Tensor     # (B, P)
    selected_count: int = None
    current_node: torch.Tensor = None    # (B, P)
    state_mask: torch.Tensor = None      # (B, P, N)
    finished: torch.Tensor = None         # (B, P)

class ATSPEnv:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.pomo_size = env_params['pomo_size']
        self.node_num = env_params['node_num']

        # View
        self.problems = None
        self.trans_problems = None
        self.node_coords = None

        # Const
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None

        # Dynamic
        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None
        self.visited_flag = None
        self.mask = None
        self.finished = None

        self.step_state = None

    def load_problems(self, batch_size, aug_enable=False, aug_factor=1):
        self.batch_size = batch_size
        problem_gen_params = self.env_params['problem_gen_params']

        self.problems, self.trans_problems, self.node_coords = \
            get_random_problems(self.batch_size, self.node_num, problem_gen_params)

        if aug_enable:
            self.problems = self.problems.repeat(aug_factor, 1, 1)
            self.trans_problems = self.trans_problems.repeat(aug_factor, 1, 1)
            self.node_coords = self.node_coords.repeat(aug_factor, 1, 1)
            self.batch_size = self.batch_size * aug_factor

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def load_equivalent_view(self):
        self.problems = self.trans_problems

    def load_problems_manual(self, problems):
        self.batch_size = problems.size(0)
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        self.problems = problems

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.empty((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        self.visited_flag = torch.zeros((self.batch_size, self.pomo_size, self.node_num))
        self.mask = torch.zeros((self.batch_size, self.pomo_size, self.node_num))
        self.finished = torch.zeros((self.batch_size, self.pomo_size), dtype=torch.bool)

        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)

        reward = None
        done = False
        return Reset_State(problems=self.problems), reward, done

    def pre_step(self):
        reward = None
        done = False
        self._update_step_state()
        return self.step_state, reward, done

    def step(self, selected):
        self.selected_count += 1
        self.current_node = selected

        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        self.visited_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        self.mask = self.visited_flag.clone()

        self._update_step_state()

        done = (self.selected_count == self.node_num)
        reward = -self._get_total_distance() if done else None

        return self.step_state, reward, done

    def _update_step_state(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.state_mask = self.mask
        self.step_state.finished = self.finished

    def _get_total_distance(self):
        n_from = self.selected_node_list
        n_to = self.selected_node_list.roll(dims=2, shifts=-1)
        B_idx = self.BATCH_IDX[:, :, None].expand(self.batch_size, self.pomo_size, self.selected_count)

        cost = self.problems[B_idx, n_from, n_to]
        total = cost.sum(2)
        return total