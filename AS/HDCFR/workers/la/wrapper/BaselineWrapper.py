# Copyright (c) Eric Steinberger 2020

import torch

from PokerRL.rl import rl_util
from PokerRL.rl.neural.NetWrapperBase import NetWrapperBase, NetWrapperArgsBase

from HDCFR.workers.la.neural.BaselineNet import BaselineNet

class BaselineWrapper(NetWrapperBase):

    def __init__(self, env_bldr, baseline_args, device): # why the device is different from the one in the args
        super().__init__(
            net=BaselineNet(env_bldr=env_bldr, args=baseline_args.net_args, device=device),
            owner=None,
            env_bldr=env_bldr,
            args=baseline_args,
            device=device,
        )
        self.dim_c = baseline_args.dim_c

        # self._batch_arranged = torch.arange(self._args.batch_size, dtype=torch.long, device=self.device)
        # self._minus_e20 = torch.full((self._args.batch_size, self._env_bldr.N_ACTIONS,),
        #                              fill_value=-10e20,
        #                              device=self.device,
        #                              dtype=torch.float32,
        #                              requires_grad=False)


    def get_b(self, pub_obses, range_idxs, option_idxs, legal_actions_lists, to_np=False):
        with torch.no_grad():
            range_idxs = torch.tensor(range_idxs, dtype=torch.long, device=self.device)
            option_idxs = torch.tensor(option_idxs, dtype=torch.long, device=self.device)

            masks = rl_util.batch_get_legal_action_mask_torch(n_actions=self._env_bldr.N_ACTIONS,
                                                              legal_actions_lists=legal_actions_lists,
                                                              device=self.device, dtype=torch.float32)
            self.eval()
            q = self._net(pub_obses=pub_obses, range_idxs=range_idxs, option_idxs=option_idxs,
                          legal_action_masks=masks)
            q *= masks

            if to_np:
                q = q.cpu().numpy()

            return q

    def _mini_batch_loop(self, buffer, grad_mngr):
        batch_pub_obs, \
        batch_range_idx, \
        batch_cur_option, \
        batch_legal_action_mask, \
        batch_a, \
        batch_r, \
        batch_pub_obs_tp1, \
        batch_legal_action_mask_tp1, \
        batch_done, \
        batch_strat_tp1 = buffer.sample(device=self.device, batch_size=self._args.batch_size)

        # [batch_size, n_actions]
        q1_t = self._net(pub_obses=batch_pub_obs, range_idxs=batch_range_idx, option_idxs=batch_cur_option,
                         legal_action_masks=batch_legal_action_mask.to(torch.float32))

        batch_pub_obs_tp1 = batch_pub_obs_tp1.unsqueeze(1).repeat(1, self.dim_c, 1).view(-1, batch_pub_obs_tp1.shape[-1])
        batch_range_idx = batch_range_idx.unsqueeze(1).repeat(1, self.dim_c).view(-1)
        batch_legal_action_mask_tp1 = \
            batch_legal_action_mask_tp1.unsqueeze(1).repeat(1, self.dim_c, 1).view(-1, batch_legal_action_mask_tp1.shape[-1])
        batch_all_options = torch.tensor(list(range(self.dim_c)) * q1_t.shape[0], dtype=batch_cur_option.dtype, device=batch_cur_option.device)

        q1_tp1 = self._net(pub_obses=batch_pub_obs_tp1, range_idxs=batch_range_idx, option_idxs=batch_all_options,
                           legal_action_masks=batch_legal_action_mask_tp1.to(torch.float32)).detach()

        _minus_e20 = torch.full(q1_tp1.shape, fill_value=-10e20, device=self.device, dtype=torch.float32, requires_grad=False)
        q1_tp1 = torch.where(batch_legal_action_mask_tp1, q1_tp1, _minus_e20).view(q1_t.shape[0], self.dim_c, -1).view(q1_t.shape[0], -1)

        # print(q1_tp1.shape, batch_strat_tp1.shape)

        q_tp1_of_atp1 = (q1_tp1 * batch_strat_tp1.view(q1_t.shape[0], -1)).sum(-1)
        q_tp1_of_atp1 *= (1.0 - batch_done)
        target = batch_r + q_tp1_of_atp1


        # ______________________________________________ TD Learning _______________________________________________
        # [batch_size]
        _batch_arranged = torch.arange(batch_a.shape[0], dtype=torch.long, device=self.device)
        q1_t_of_a_selected = q1_t[_batch_arranged, batch_a]
        # print("1: ", q1_t.shape, q1_t_of_a_selected.shape)
        grad_mngr.backprop(pred=q1_t_of_a_selected, target=target)
        # print("2: ", self._args.loss_str)


class BaselineTrainingArgs(NetWrapperArgsBase):

    def __init__(self,
                 net_args,
                 max_buffer_size=2e5,
                 n_batches_per_iter_baseline=500,
                 dim_c=3,
                 batch_size=512,
                 optim_str="adam",
                 loss_str="hdcfr_baseline_loss",
                 lr=0.001,
                 grad_norm_clipping=1.0,
                 device_training="cpu",
                 init_model="last"
                 ):
        super().__init__(batch_size=batch_size,
                         optim_str=optim_str,
                         loss_str=loss_str,
                         lr=lr,
                         grad_norm_clipping=grad_norm_clipping,
                         device_training=device_training)
        self.net_args = net_args
        self.max_buffer_size = int(max_buffer_size)
        self.n_batches_per_iter_baseline = n_batches_per_iter_baseline
        self.init_model = init_model
        self.dim_c = dim_c
