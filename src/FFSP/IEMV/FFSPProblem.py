import torch

import torch

def get_random_problems(batch_size, stage_cnt, machine_cnt_list, job_cnt, process_time_params):
    time_low_list = process_time_params['time_low_list']
    time_high_list = process_time_params['time_high_list']

    problems = []
    for stage_num in range(stage_cnt):
        time_low = time_low_list[stage_num]
        time_high = time_high_list[stage_num]
        machine_cnt = machine_cnt_list[stage_num]
        stage_prob = torch.randint(low=time_low, high=time_high, size=(batch_size, job_cnt, machine_cnt))
        problems.append(stage_prob)

    flip_problems = problems[::-1]

    return problems, flip_problems

