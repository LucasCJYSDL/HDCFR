import torch
import torch.nn as nn


class LowAvrgNet(nn.Module):

    def __init__(self, avrg_net_args, env_bldr, device):
        super().__init__()
        self.args = avrg_net_args
        self.env_bldr = env_bldr
        self.n_actions = self.env_bldr.N_ACTIONS
        self._n_options = avrg_net_args.mpm_args.dim_c

        MPM = avrg_net_args.mpm_args.get_mpm_cls()

        self._mpm = MPM(env_bldr=env_bldr, device=device, mpm_args=self.args.mpm_args)

        self._head = nn.Linear(in_features=self._mpm.output_units, out_features=self.n_actions * self._n_options)

        self.to(device)

    def set_option_emb(self, option_emb):
        self._mpm.set_option_emb(option_emb) # danger

    def forward(self, pub_obses, range_idxs, option_idxs, legal_action_masks):
        """
        Softmax is not applied in here! It is separate in training and action fns
        """

        y = self._mpm(pub_obses=pub_obses, range_idxs=range_idxs, option_idxs=option_idxs)
        y = self._head(y)

        y = y.view(-1, self._n_options, self.n_actions)
        ind = option_idxs.unsqueeze(-1).view(-1, 1, 1).expand(-1, 1, self.n_actions)
        y = y.gather(dim=-2, index=ind).squeeze(dim=-2)  # (bs, dim_a)

        y = torch.where(legal_action_masks == 1,
                        y,
                        torch.FloatTensor([-10e20]).to(device=y.device).expand_as(y))
        return y


class LowAvrgArgs:

    def __init__(self,
                 mpm_args,
                 n_units_final
                 ):
        self.mpm_args = mpm_args
        self.n_units_final = n_units_final