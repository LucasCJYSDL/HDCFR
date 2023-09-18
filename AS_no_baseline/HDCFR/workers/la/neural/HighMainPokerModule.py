import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PokerRL.rl.neural.CardEmbedding import CardEmbedding
from PokerRL.rl.neural.LayerNorm import LayerNorm
from HDCFR.workers.la.neural.MHA_models import SkillPolicy, layer_init, range_tensor


class HighMainPokerModule(nn.Module):

    def __init__(self,
                 env_bldr,
                 device,
                 mpm_args,
                 ):
        super().__init__()

        self._args = mpm_args
        self._env_bldr = env_bldr

        self._device = device

        self._board_start = self._env_bldr.obs_board_idxs[0]
        self._board_stop = self._board_start + len(self._env_bldr.obs_board_idxs)

        self.card_emb = CardEmbedding(env_bldr=env_bldr, dim=mpm_args.dim_h, device=device)

        self.dim_s = self.card_emb.out_size + self._env_bldr.pub_obs_size - self._env_bldr.obs_size_board
        self.dim_c = self._args.dim_c
        # MHA

        # embedding matrix and the kernel matrix of the attention module
        # the last one corresponds to the initial option choice
        self.embed_option = nn.Embedding(self._args.dim_c + 1, self._args.dmodel)
        nn.init.orthogonal(self.embed_option.weight)  # initialization

        self.de_state_lc = layer_init(nn.Linear(self.dim_s, self._args.dmodel))
        self.de_state_norm = nn.LayerNorm(self._args.dmodel)
        self.doe = SkillPolicy(self._args.dmodel, self._args.mha_nhead,
                               self._args.mha_nlayers, self._args.mha_nhid, self._args.mha_dropout)
        self.de_logtis_lc_1 = layer_init(nn.Linear(2 * self._args.dmodel, self._args.dim_h))
        self.de_logtis_lc_2 = layer_init(nn.Linear(self._args.dim_h, self._args.dim_h))

        self.norm = LayerNorm(self._args.dim_h)

        self.to(device)

    @property
    def output_units(self):
        return self._args.dim_h

    @property
    def device(self):
        return self._device

    def get_option_emb(self):
        return self.embed_option

    # TODO: can be optimized by specifying the option choices
    def forward(self, pub_obses, range_idxs):

        # betting history, [bs, 57]
        if isinstance(pub_obses, list):
            pub_obses = torch.from_numpy(np.array(pub_obses)).to(self._device, torch.float32)
        hist_o = torch.cat([
            pub_obses[:, :self._board_start],
            pub_obses[:, self._board_stop:]
        ], dim=-1)
        # Card embeddings, [bs, 128]
        card_o = self.card_emb(pub_obses=pub_obses, range_idxs=range_idxs) # embedding of the public and private cards
        # state
        s = torch.cat([hist_o, card_o], dim=-1) # TODO: separately process the two parts in the state
        bs = s.shape[0]
        assert s.shape[1] == self.dim_s

        # option embeddings
        # embed_all_idx: [dim_c+1, bs]
        embed_all_idx = range_tensor(self.dim_c+1, self.device).repeat(bs, 1).t()
        wt = self.embed_option(embed_all_idx)  # (dim_c+1, bs, dim_e) # this is the attention kernel matrix

        # state input
        s_rep = s.unsqueeze(1).repeat(1, self.dim_c+1, 1).view(-1, self.dim_s)  # (bs*(dim_c+1), dim_s)
        s_hat = F.relu(self.de_state_lc(s_rep))  # (bs*(dim_c+1), dim_e)
        s_hat = self.de_state_norm(s_hat)  # (bs*(dim_c+1), dim_e)

        # option input
        embed_all_idx = range_tensor(self.dim_c+1, self.device).repeat(bs, 1)  # (bs, dim_c+1)
        prev_options = embed_all_idx.view(-1, 1)  # (bs*(dim_c+1), 1)
        ct_1 = self.embed_option(prev_options.t()).detach()  # (1, bs*(dim_c+1), dim_e)

        # concat
        opt_cat_1 = torch.cat([s_hat.unsqueeze(0), ct_1], dim=0)  # (2, bs*(dim_c+1), dim_e)
        rdt = self.doe(wt, opt_cat_1)  # (2, bs*(dim_c+1), dim_e)
        dt = torch.cat([rdt[0].squeeze(0), rdt[1].squeeze(0)], dim=-1)  # (bs*(dim_c+1), 2*dim_e)

        # post process
        opt_logits = F.relu(self.de_logtis_lc_1(dt))  # (bs*(dim_c+1), dim_h)
        opt_logits = F.relu(self.de_logtis_lc_2(opt_logits) + opt_logits) # (bs*(dim_c+1), dim_h) # TODO: with the skip or not

        opt_logits = self.norm(opt_logits) # (bs*(dim_c+1), dim_h) # TODO: with the norm or not

        # opt_logits = opt_logits.view(bs, self.dim_c+1, self.dim_h) # used in the wrapper

        return opt_logits


class HighMPMArgs:
    # TODO: normalize the last layer output
    def __init__(self,
                 dim_h,
                 dim_c,
                 dmodel,
                 mha_nhead,
                 mha_nlayers,
                 mha_nhid,
                 mha_dropout
                 ):
        self.dim_h = dim_h # 64
        self.dim_c = dim_c # 3
        self.dmodel = dmodel # important, 64
        self.mha_nhead = mha_nhead # important, 1
        self.mha_nlayers = mha_nlayers # important, 1
        self.mha_nhid = mha_nhid # 64
        self.mha_dropout = mha_dropout # 0.2

    def get_mpm_cls(self):
        return HighMainPokerModule