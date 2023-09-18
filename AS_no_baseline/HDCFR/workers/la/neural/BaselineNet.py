import torch.nn as nn
from torch.nn import functional as F
from HDCFR.workers.la.neural.MHA_models import layer_init_zero

class BaselineNet(nn.Module):

    def __init__(self, env_bldr, args, device):
        super().__init__()

        self._env_bldr = env_bldr
        self._args = args
        self._n_actions = env_bldr.N_ACTIONS
        self._n_options = args.mpm_args.dim_c

        MPM = args.mpm_args.get_mpm_cls()
        self._mpm = MPM(env_bldr=env_bldr, device=device, mpm_args=args.mpm_args)

        # ____________________ advantage net & v net layers _______________________
        self._adv_layer = nn.Linear(in_features=self._mpm.output_units, out_features=args.n_units_final)
        self._adv = nn.Linear(in_features=args.n_units_final, out_features=self._n_actions * self._n_options)

        self._adv = layer_init_zero(nn.Linear(in_features=args.n_units_final,
                                              out_features=self._n_actions * self._n_options))

        self._state_v_layer = nn.Linear(in_features=self._mpm.output_units, out_features=args.n_units_final)
        self._v = nn.Linear(in_features=args.n_units_final, out_features=self._n_options)

        self._v = layer_init_zero(nn.Linear(in_features=args.n_units_final, out_features=self._n_options))

        self.to(device)

    def forward(self, pub_obses, range_idxs, option_idxs, legal_action_masks):

        shared_out = self._mpm(pub_obses=pub_obses, range_idxs=range_idxs)

        adv = self._get_adv(shared_out=shared_out, option_idxs=option_idxs, legal_action_masks=legal_action_masks)

        val = F.relu(self._state_v_layer(shared_out))
        val = self._v(val) # (bs, dim_c)
        val = val.view(-1, self._n_options, 1)
        ind = option_idxs.unsqueeze(-1).view(-1, 1, 1)
        # print("1: ", val.shape, ind.shape)
        val = val.gather(dim=-2, index=ind).squeeze(dim=-2)
        # print("1: ", val.shape, ind.shape)
        val = val.expand_as(adv)

        # print("2: ", adv.shape, val.shape)
        return (val + adv) * legal_action_masks

    def _get_adv(self, shared_out, option_idxs, legal_action_masks):
        y = F.relu(self._adv_layer(shared_out))
        y = self._adv(y) # (bs, dim_c * dim_a)
        y = y.view(-1, self._n_options, self._n_actions)
        ind = option_idxs.unsqueeze(-1).view(-1, 1, 1).expand(-1, 1, self._n_actions)
        y = y.gather(dim=-2, index=ind).squeeze(dim=-2) # (bs, dim_a)
        # sets all illegal actions to 0 for mean computation
        y *= legal_action_masks

        # can't directly compute mean cause illegal actions are still in there. Computing sum and dividing by ||A(I)||
        mean = (y.sum(dim=1) / legal_action_masks.sum(dim=1)).unsqueeze(1).expand(-1, self._n_actions)
        # print("1: ", y.shape, mean.shape, legal_action_masks.shape)
        # subtracting mean also subtracts from illegal actions; have to mask again
        return (y - mean) * legal_action_masks


class BaselineArgs:

    def __init__(self, n_units_final, mpm_args):
        self.n_units_final = n_units_final
        self.mpm_args = mpm_args
