# Copyright (c) Eric Steinberger 2020

import torch

from PokerRL.rl.neural.NetWrapperBase import NetWrapperArgsBase as _NetWrapperArgsBase
from PokerRL.rl.neural.NetWrapperBase import NetWrapperBase as _NetWrapperBase

from HDCFR.workers.la.neural.HighAdvNet import HighAdvVet
from HDCFR.workers.la.neural.LowAdvNet import LowAdvNet


class AdvWrapper(_NetWrapperBase):

    def __init__(self, env_bldr, adv_training_args, owner, device, is_high):
        self.is_high = is_high

        if is_high:
            net_obj = HighAdvVet(env_bldr=env_bldr, args=adv_training_args.high_adv_net_args, device=device)
        else:
            net_obj = LowAdvNet(env_bldr=env_bldr, args=adv_training_args.low_adv_net_args, device=device)

        super().__init__(
            net=net_obj,
            env_bldr=env_bldr,
            args=adv_training_args,
            owner=owner,
            device=device
        )


    def get_advantages(self, pub_obses, range_idxs, option_idxs, legal_action_mask=None):
        self._net.eval()
        with torch.no_grad():
            if self.is_high:
                return self._net(pub_obses=pub_obses, range_idxs=range_idxs, option_idxs=option_idxs)
            else:
                assert legal_action_mask
                return self._net(pub_obses=pub_obses, range_idxs=range_idxs, option_idxs=option_idxs,
                                 legal_action_masks=legal_action_mask)


    def _mini_batch_loop(self, buffer, grad_mngr):
        batch_pub_obs, \
        batch_range_idxs, \
        batch_last_option, \
        batch_cur_option, \
        batch_legal_action_masks, \
        batch_high_adv, \
        batch_low_adv, \
        batch_high_loss_weight, batch_low_loss_weight = buffer.sample(device=self.device, batch_size=self._args.batch_size) # important template

        # decoupling the training data for the high/low networks
        if self.is_high:
            # [batch_size, n_actions]
            high_adv_pred = self._net(pub_obses=batch_pub_obs,
                                      range_idxs=batch_range_idxs,
                                      option_idxs=batch_last_option)
            # if loss weight equals to the time step, then this is linear CFR
            grad_mngr.backprop(pred=high_adv_pred, target=batch_high_adv,
                               loss_weights=batch_high_loss_weight.unsqueeze(-1).expand_as(batch_high_adv))
        else:
            low_adv_pred = self._net(pub_obses=batch_pub_obs,
                                      range_idxs=batch_range_idxs,
                                      option_idxs=batch_cur_option,
                                      legal_action_masks=batch_legal_action_masks)
            # for the illegal actions the low adv value has to be 0
            grad_mngr.backprop(pred=low_adv_pred, target=batch_low_adv,
                               loss_weights=batch_low_loss_weight.unsqueeze(-1).expand_as(batch_low_adv))


class HierAdvWrapper:
    def __init__(self, env_bldr, adv_training_args, owner, device):
        self.high_adv = AdvWrapper(env_bldr, adv_training_args, owner, device, is_high=True)
        self.low_adv = AdvWrapper(env_bldr, adv_training_args, owner, device, is_high=False)
        self.low_adv._net.set_option_emb(self.high_adv._net.get_option_emb()) # danger, check
        self.device = device

    def get_option_emb(self):
        return self.high_adv._net.get_option_emb()

    def get_high_advantages(self, pub_obses, range_idxs, option_idxs):
        return self.high_adv.get_advantages(pub_obses, range_idxs, option_idxs)

    def get_low_advantages(self, pub_obses, range_idxs, option_idxs, legal_action_mask):
        return self.low_adv.get_advantages(pub_obses, range_idxs, option_idxs, legal_action_mask)

    def get_high_grads(self, buffer):
        return self.high_adv.get_grads_one_batch_from_buffer(buffer)

    def get_low_grads(self, buffer):
        return self.low_adv.get_grads_one_batch_from_buffer(buffer)

    def get_loss_last_batch(self):
        return (self.high_adv.loss_last_batch, self.low_adv.loss_last_batch)

    def load_net_state_dict(self, state_dict):
        self.high_adv.load_net_state_dict(state_dict[0])
        self.low_adv.load_net_state_dict(state_dict[1])

    def net_state_dict(self):
        return (self.high_adv.net_state_dict(), self.low_adv.net_state_dict())


class HierAdvTrainingArgs(_NetWrapperArgsBase):

    def __init__(self,
                 high_adv_net_args,
                 low_adv_net_args,
                 n_batches_adv_training=1000,
                 batch_size=4096,
                 optim_str="adam",
                 loss_str="weighted_mse",
                 lr=0.001,
                 grad_norm_clipping=10.0,
                 device_training="cpu",
                 max_buffer_size=2e6,
                 lr_patience=100,
                 init_adv_model="last",
                 ):
        super().__init__(batch_size=batch_size,
                         optim_str=optim_str, loss_str=loss_str, lr=lr, grad_norm_clipping=grad_norm_clipping,
                         device_training=device_training)
        self.high_adv_net_args = high_adv_net_args
        self.low_adv_net_args = low_adv_net_args
        self.n_batches_adv_training = n_batches_adv_training
        self.lr_patience = lr_patience
        self.max_buffer_size = int(max_buffer_size)
        self.init_adv_model = init_adv_model
