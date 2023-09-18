# Copyright (c) Eric Steinberger 2020

import torch
import torch.nn.functional as nnf

from PokerRL.rl import rl_util
from PokerRL.rl.neural.NetWrapperBase import NetWrapperArgsBase as _NetWrapperArgsBase
from PokerRL.rl.neural.NetWrapperBase import NetWrapperBase as _NetWrapperBase

from HDCFR.workers.la.neural.HighAvrgNet import HighAvrgNet
from HDCFR.workers.la.neural.LowAvrgNet import LowAvrgNet


class AvrgWrapper(_NetWrapperBase):

    def __init__(self, owner, env_bldr, avrg_training_args, device, is_high):

        self.is_high = is_high
        if is_high:
            net_obj = HighAvrgNet(avrg_net_args=avrg_training_args.high_avrg_net_args, env_bldr=env_bldr, device=device)
        else:
            net_obj = LowAvrgNet(avrg_net_args=avrg_training_args.low_avrg_net_args, env_bldr=env_bldr, device=device)

        super().__init__(
            net=net_obj,
            env_bldr=env_bldr,
            args=avrg_training_args,
            owner=owner,
            device=device
        )
        self._all_range_idxs = torch.arange(self._env_bldr.rules.RANGE_SIZE, device=self.device, dtype=torch.long)

    def get_a_probs(self, pub_obses, range_idxs, option_idxs, legal_actions_lists):
        # range_idxs (np.ndarray):    array of range_idxs (one for each pub_obs) tensor([2, 421, 58, 912, ...])
        assert not self.is_high
        with torch.no_grad():
            masks = rl_util.batch_get_legal_action_mask_torch(n_actions=self._env_bldr.N_ACTIONS,
                                                              legal_actions_lists=legal_actions_lists,
                                                              device=self.device)
            masks = masks.view(1, -1)

            pred = self._net(pub_obses=pub_obses,
                             range_idxs=torch.from_numpy(range_idxs).to(dtype=torch.long, device=self.device),
                             option_idxs=torch.from_numpy(option_idxs).to(dtype=torch.long, device=self.device),
                             legal_action_masks=masks)
            return nnf.softmax(pred, dim=-1).cpu().numpy()

    def get_z_probs(self, pub_obses, range_idxs, option_idxs):
        assert self.is_high
        with torch.no_grad():
            pred = self._net(pub_obses=pub_obses,
                             range_idxs=torch.from_numpy(range_idxs).to(dtype=torch.long, device=self.device),
                             option_idxs=torch.from_numpy(option_idxs).to(dtype=torch.long, device=self.device))
            return nnf.softmax(pred, dim=-1).cpu().numpy()


    def _mini_batch_loop(self, buffer, grad_mngr): # main component, specific for each network agent
        batch_pub_obs, \
        batch_range_idxs, \
        batch_last_option, \
        batch_cur_option, \
        batch_legal_action_masks, \
        batch_z_probs, \
        batch_a_probs, \
        batch_high_loss_weight, batch_low_loss_weight = buffer.sample(device=self.device, batch_size=self._args.batch_size)
        # print("1: ", batch_pub_obs.shape)

        if not self.is_high:
            # [batch_size, n_actions]
            strat_pred = self._net(pub_obses=batch_pub_obs,
                                   range_idxs=batch_range_idxs,
                                   option_idxs=batch_cur_option,
                                   legal_action_masks=batch_legal_action_masks)
            strat_pred = nnf.softmax(strat_pred, dim=-1)
            grad_mngr.backprop(pred=strat_pred, target=batch_a_probs,
                               loss_weights=batch_low_loss_weight.unsqueeze(-1).expand_as(batch_a_probs))
        else:
            strat_pred = self._net(pub_obses=batch_pub_obs,
                                   range_idxs=batch_range_idxs,
                                   option_idxs=batch_last_option)
            strat_pred = nnf.softmax(strat_pred, dim=-1)
            grad_mngr.backprop(pred=strat_pred, target=batch_z_probs,
                               loss_weights=batch_high_loss_weight.unsqueeze(-1).expand_as(batch_z_probs))


class HierAvrgWrapper:
    def __init__(self, owner, env_bldr, avrg_training_args, device):
        self.high_avrg = AvrgWrapper(owner, env_bldr, avrg_training_args, device, is_high=True)
        self.low_avrg = AvrgWrapper(owner, env_bldr, avrg_training_args, device, is_high=False)
        self.low_avrg._net.set_option_emb(self.high_avrg._net.get_option_emb())
        self.device = device

    def get_z_probs(self, pub_obses, range_idxs, option_idxs):
        return self.high_avrg.get_z_probs(pub_obses, range_idxs, option_idxs)

    def get_a_probs(self, pub_obses, range_idxs, option_idxs, legal_actions_lists):
        return self.low_avrg.get_a_probs(pub_obses, range_idxs, option_idxs, legal_actions_lists)

    def get_high_grads(self, buffer):
        return self.high_avrg.get_grads_one_batch_from_buffer(buffer)

    def get_low_grads(self, buffer):
        return self.low_avrg.get_grads_one_batch_from_buffer(buffer)

    def get_loss_last_batch(self):
        return (self.high_avrg.loss_last_batch, self.low_avrg.loss_last_batch)

    def eval(self):
        self.high_avrg.eval()
        self.low_avrg.eval()

    def net_state_dict(self):
        return (self.high_avrg.net_state_dict(), self.low_avrg.net_state_dict())

    def load_net_state_dict(self, state_dict):
        self.high_avrg.load_net_state_dict(state_dict[0])
        self.low_avrg.load_net_state_dict(state_dict[1])

    def get_a_probs_for_each_hand(self, pub_obs, legal_actions_list, option_idx):
        with torch.no_grad():
            if option_idx is None: # the first iteration
                option_idx = torch.tensor([self.high_avrg._net.n_options for _ in range(self.high_avrg._env_bldr.rules.RANGE_SIZE)],
                                          dtype=torch.long, device=self.high_avrg.device)

            mask = rl_util.get_legal_action_mask_torch(n_actions=self.high_avrg._env_bldr.N_ACTIONS,
                                                       legal_actions_list=legal_actions_list,
                                                       device=self.high_avrg.device, dtype=torch.uint8)
            # print("1: ", legal_actions_list, mask)
            mask = mask.unsqueeze(0).expand(self.high_avrg._env_bldr.rules.RANGE_SIZE, -1)
            # print("2: ", mask.shape, self.high_avrg._all_range_idxs)
            pub_obses = [pub_obs] * self.high_avrg._env_bldr.rules.RANGE_SIZE

            # TODO: compute the action prob by synthesizing the all the possibilities of the option choice
            # get the high-level strategy
            z_pred = self.high_avrg._net(pub_obses=pub_obses,
                                         range_idxs=self.high_avrg._all_range_idxs,
                                         option_idxs=option_idx)
            z_prob = nnf.softmax(z_pred, dim=-1)
            cur_options = torch.multinomial(z_prob, num_samples=1).squeeze().detach().clone() # (bs, )
            # print("3: ", z_prob, cur_options, self.low_avrg._all_range_idxs)
            # get the low-level strategy
            a_pred = self.low_avrg._net(pub_obses=pub_obses,
                                        range_idxs=self.low_avrg._all_range_idxs,
                                        option_idxs=cur_options.to(dtype=torch.long, device=self.low_avrg.device),
                                        legal_action_masks=mask)

            return nnf.softmax(a_pred, dim=1).cpu().numpy(), cur_options








class HierAvrgTrainingArgs(_NetWrapperArgsBase):

    def __init__(self,
                 high_avrg_net_args,
                 low_avrg_net_args,
                 n_batches_avrg_training=1000,
                 batch_size=4096,
                 optim_str="adam",
                 loss_str="weighted_mse",
                 lr=0.001,
                 grad_norm_clipping=10.0,
                 device_training="cpu",
                 max_buffer_size=2e6,
                 lr_patience=100,
                 init_avrg_model="random",
                 ):
        super().__init__(batch_size=batch_size,
                         optim_str=optim_str,
                         loss_str=loss_str,
                         lr=lr,
                         grad_norm_clipping=grad_norm_clipping,
                         device_training=device_training)

        self.high_avrg_net_args = high_avrg_net_args
        self.low_avrg_net_args = low_avrg_net_args
        self.n_batches_avrg_training = n_batches_avrg_training
        self.max_buffer_size = int(max_buffer_size)
        self.lr_patience = lr_patience
        self.init_avrg_model = init_avrg_model
