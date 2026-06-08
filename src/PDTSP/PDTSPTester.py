import torch
from logging import getLogger

from PDTSPEnv import PDTSPEnv as Env
from PDTSPModel import PDTSPModel as Model
from utils import get_result_folder, AverageMeter, TimeEstimator


class PDTSPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        self.logger = getLogger(name='tester')
        self.result_folder = get_result_folder()

        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            device = torch.device('cuda', cuda_device_num)
        else:
            device = torch.device("cpu")
        torch.set_default_device(device)

        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        model_load = self.tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.time_estimator = TimeEstimator()

    def run(self):
        score = AverageMeter()
        aug_score = AverageMeter()
        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)
            batch_score, batch_aug_score = self._test_one_batch(batch_size)
            score.update(batch_score, batch_size)
            aug_score.update(batch_aug_score, batch_size)
            episode += batch_size

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score.avg, aug_score.avg))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score.avg))

    def _test_one_batch(self, batch_size):
        aug_enable = self.tester_params['augmentation_enable']
        aug_factor = self.tester_params['aug_factor'] if aug_enable else 1

        self.model.eval()
        with torch.no_grad():
            # Main view
            self.env.load_problems(batch_size, aug_enable=aug_enable, aug_factor=aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            state, reward, done = self.env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = self.env.step(selected)

            aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)

            # Base score: no aug, single view, best pomo
            main_no_aug_best_pomo, _ = aug_reward[0, :, :].max(dim=1)
            score_base = -main_no_aug_best_pomo.float().mean()

            # Best of main view: all aug & pomo
            main_all_best, _ = aug_reward.max(dim=2)
            main_all_best, _ = main_all_best.max(dim=0)

            # Equivalent view
            self.env.load_equivalent_view()
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            state, reward, done = self.env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = self.env.step(selected)

            aug_reward_equiv = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
            equiv_all_best, _ = aug_reward_equiv.max(dim=2)
            equiv_all_best, _ = equiv_all_best.max(dim=0)

            # Global best: all aug, all view, all pomo
            final_all_best = torch.max(main_all_best, equiv_all_best)
            score_global_best = -final_all_best.float().mean()

        return score_base.item(), score_global_best.item()