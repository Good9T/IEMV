import torch


def get_random_problems(batch_size, node_num, problem_gen_params):
    scaler = problem_gen_params['scaler']
    perturb_min = problem_gen_params.get('perturb_min', 0.1)
    perturb_max = problem_gen_params.get('perturb_max', 0.5)
    asym_bias = problem_gen_params.get('asym_bias', 0.2)

    node_coords = torch.rand(batch_size, node_num, 2)
    coord_diff = node_coords.unsqueeze(2) - node_coords.unsqueeze(1)
    straight_dist = torch.norm(coord_diff, p=2, dim=-1)

    perturb_ratio = torch.rand(batch_size, node_num, node_num) * (perturb_max - perturb_min) + perturb_min
    asym_bias_matrix = torch.rand(batch_size, node_num, node_num) * asym_bias
    asym_bias_matrix = asym_bias_matrix - asym_bias_matrix.transpose(1, 2) / 2

    problems_distance = straight_dist * (1 + perturb_ratio + asym_bias_matrix)
    problems_distance = torch.max(problems_distance, straight_dist)
    problems_distance[:, torch.arange(node_num), torch.arange(node_num)] = 0.0

    while True:
        old_problems_distance = problems_distance.clone()
        problems_distance, _ = (
                    problems_distance[:, :, None, :] + problems_distance[:, None, :, :].transpose(2, 3)).min(dim=3)
        if (problems_distance == old_problems_distance).all():
            break

    problems = problems_distance.float() / scaler
    trans_problems = torch.transpose(problems, 1, 2)

    return problems, trans_problems, node_coords