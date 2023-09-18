import torch.nn as nn

class HighAvrgNet(nn.Module):

    def __init__(self, avrg_net_args, env_bldr, device):
        super().__init__()

        self.args = avrg_net_args
        self.env_bldr = env_bldr
        self.n_options = avrg_net_args.mpm_args.dim_c

        MPM = avrg_net_args.mpm_args.get_mpm_cls()

        self._mpm = MPM(env_bldr=env_bldr, device=device, mpm_args=self.args.mpm_args)

        self._head = nn.Linear(in_features=self._mpm.output_units, out_features=self.n_options)

        self.to(device)

    def get_option_emb(self):
        return self._mpm.get_option_emb()

    def forward(self, pub_obses, range_idxs, option_idxs):
        """
        Softmax is not applied in here! It is adopted later.
        """
        # option_idxs: (bs, )
        y = self._mpm(pub_obses=pub_obses, range_idxs=range_idxs)
        y = y.view(-1, self.n_options + 1, self._mpm.output_units)
        # print("1: ", y.shape, y)
        y = y.gather(dim=-2, index=option_idxs.unsqueeze(-1).view(-1, 1, 1).\
                                       expand(-1, 1, self._mpm.output_units)).squeeze(dim=-2)  # (bs, dim_h)
        # print("2: ", y.shape, y, option_idxs.unsqueeze(-1).view(-1, 1, 1).expand(-1, 1, self._mpm.output_units).shape, option_idxs)

        y = self._head(y)

        return y



class HighAvrgArgs:

    def __init__(self,
                 mpm_args,
                 n_units_final
                 ):
        self.mpm_args = mpm_args
        self.n_units_final = n_units_final