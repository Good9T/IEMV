import torch
import numpy as np
import pandas as pd
import copy


def get_random_problems(batch_size, customer_size, problem_gen_params):
    scaler = problem_gen_params['scaler']
    depot_size = 1
    node_num = depot_size + customer_size * 2

    depot_xy = torch.rand(batch_size, depot_size, 2)
    pick_xy = torch.rand(batch_size, customer_size, 2)
    delivery_xy = torch.rand(batch_size, customer_size, 2)

    node_coords = torch.cat([depot_xy, pick_xy, delivery_xy], dim=1)

    coord_diff = node_coords.unsqueeze(2) - node_coords.unsqueeze(1)
    straight_dist = torch.norm(coord_diff, p=2, dim=-1)
    straight_dist[:, torch.arange(node_num), torch.arange(node_num)] = 0.0
    problems = straight_dist.float() / scaler

    node_coords_dual = torch.cat([depot_xy, delivery_xy, pick_xy], dim=1)

    coord_diff_dual = node_coords_dual.unsqueeze(2) - node_coords_dual.unsqueeze(1)
    straight_dist_dual = torch.norm(coord_diff_dual, p=2, dim=-1)
    straight_dist_dual[:, torch.arange(node_num), torch.arange(node_num)] = 0.0
    flip_problems = straight_dist_dual.float() / scaler

    return problems, flip_problems