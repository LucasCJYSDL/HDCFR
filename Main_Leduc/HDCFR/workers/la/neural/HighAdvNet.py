import torch
import torch.nn as nn
from torch.nn import functional as F
from HDCFR.workers.la.neural.MHA_models import layer_init_zero

class HighAdvVet(nn.Module):
    def __init__(self, env_bldr, args, device):
        super().__init__()

        self._env_bldr = env_bldr
        self._args = args
        self._n_options = args.mpm_args.dim_c

        MPM = args.mpm_args.get_mpm_cls()
        self._mpm = MPM(env_bldr=env_bldr, device=device, mpm_args=args.mpm_args)

        # ____________________ advantage net _______________________
        # TODO: remove v layers
        self._adv_layer = nn.Linear(in_features=self._mpm.output_units, out_features=args.n_units_final)
        self._adv = nn.Linear(in_features=args.n_units_final, out_features=self._n_options)
        # self._adv = layer_init_zero(nn.Linear(in_features=args.n_units_final, out_features=self._n_options))

        self._state_v_layer = nn.Linear(in_features=self._mpm.output_units, out_features=args.n_units_final)
        self._v = nn.Linear(in_features=args.n_units_final, out_features=1)
        # self._v = layer_init_zero(nn.Linear(in_features=args.n_units_final, out_features=1))

        self.to(device)

    def get_option_dim(self):
        return self._n_options

    def get_option_emb(self):
        return self._mpm.get_option_emb()

    def forward(self, pub_obses, range_idxs, option_idxs):
        shared_out = self._mpm(pub_obses=pub_obses, range_idxs=range_idxs)
        shared_out = shared_out.view(-1, self._n_options + 1, self._mpm.output_units)
        shared_out = shared_out.gather(dim=-2, index=option_idxs.unsqueeze(-1).view(-1, 1, 1). \
                                       expand(-1, 1, self._mpm.output_units)).squeeze(dim=-2)  # (bs, dim_h)

        adv = self._get_adv(shared_out=shared_out)

        val = F.relu(self._state_v_layer(shared_out))
        val = self._v(val).expand_as(adv)
        # print("2: ", adv.shape, val.shape)
        return val + adv

    def get_adv(self, pub_obses, range_idxs, option_idxs):
        # option_idxs: (bs, )
        shared_out = self._mpm(pub_obses=pub_obses, range_idxs=range_idxs)
        shared_out = shared_out.view(-1, self._n_options+1, self._mpm.output_units)
        shared_out = shared_out.gather(dim=-2, index=option_idxs.unsqueeze(-1).view(-1, 1, 1).\
                                       expand(-1, 1, self._mpm.output_units)).squeeze(dim=-2) # (bs, dim_h)

        return self._get_adv(shared_out=shared_out)

    def _get_adv(self, shared_out): # all options are legal
        y = F.relu(self._adv_layer(shared_out))
        y = self._adv(y)

        mean = torch.mean(y, dim=-1, keepdim=True).expand_as(y)
        # print("1: ", y.shape, mean.shape)
        return y - mean



class HighAdvArgs:

    def __init__(self, n_units_final, mpm_args):
        self.n_units_final = n_units_final
        self.mpm_args = mpm_args