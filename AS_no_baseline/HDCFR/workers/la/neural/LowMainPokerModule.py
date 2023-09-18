# Copyright (c) 2019 Eric Steinberger


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PokerRL.rl.neural.CardEmbedding import CardEmbedding
from PokerRL.rl.neural.LayerNorm import LayerNorm


class LowMainPokerModule(nn.Module):

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

        self.dropout = nn.Dropout(p=mpm_args.dropout)

        self.card_emb = CardEmbedding(env_bldr=env_bldr, dim=mpm_args.dim, device=device)

        self.embed_option = None

        if mpm_args.deep:
            self.cards_fc_1 = nn.Linear(in_features=self.card_emb.out_size, out_features=mpm_args.dim * 3)
            self.cards_fc_2 = nn.Linear(in_features=mpm_args.dim * 3, out_features=mpm_args.dim * 3)
            self.cards_fc_3 = nn.Linear(in_features=mpm_args.dim * 3, out_features=mpm_args.dim)

            self.history_1 = nn.Linear(in_features=self._env_bldr.pub_obs_size - self._env_bldr.obs_size_board,
                                       out_features=mpm_args.dim)
            self.history_2 = nn.Linear(in_features=mpm_args.dim, out_features=mpm_args.dim)

            # # TODO: remove the process layers for the option embedding
            # self.option_1 = nn.Linear(in_features=mpm_args.dmodel, out_features=mpm_args.dim)
            # self.option_2 = nn.Linear(in_features=mpm_args.dim, out_features=mpm_args.dim)

            self.comb_1 = nn.Linear(in_features=2 * mpm_args.dim, out_features=mpm_args.dim)
            self.comb_2 = nn.Linear(in_features=mpm_args.dim, out_features=mpm_args.dim)

        else:
            self.layer_1 = nn.Linear(in_features=self.card_emb.out_size + self._env_bldr.pub_obs_size
                                                 - self._env_bldr.obs_size_board + mpm_args.dmodel,
                                     out_features=mpm_args.dim)
            self.layer_2 = nn.Linear(in_features=mpm_args.dim, out_features=mpm_args.dim)
            self.layer_3 = nn.Linear(in_features=mpm_args.dim, out_features=mpm_args.dim)

        if self._args.normalize:
            self.norm = LayerNorm(mpm_args.dim)

        self.to(device)
        # print("n parameters:", sum(p.numel() for p in self.parameters() if p.requires_grad))

    @property
    def output_units(self):
        return self._args.dim

    @property
    def device(self):
        return self._device

    def set_option_emb(self, option_emb):
        self.embed_option = option_emb # danger

    def forward(self, pub_obses, range_idxs, option_idxs):

        if isinstance(pub_obses, list):
            pub_obses = torch.from_numpy(np.array(pub_obses)).to(self._device, torch.float32)

        # public history (bs, 57)
        hist_o = torch.cat([
            pub_obses[:, :self._board_start],
            pub_obses[:, self._board_stop:]
        ], dim=-1)

        # card embeddings, (bs, 128)
        card_o = self.card_emb(pub_obses=pub_obses, range_idxs=range_idxs) # embedding of the public and private cards

        # option embeddings
        # assert self.embed_option
        # ct = option_idxs.to(self._device) # should be torch long, (bs, )
        # ct_o = self.embed_option(ct.unsqueeze(0)).detach().squeeze(0)  # (bs, dim_e)
        # print("1: ", ct.shape, ct_o.shape)
        # Network
        if self._args.dropout > 0:
            A = lambda x: self.dropout(F.relu(x))
        else:
            A = lambda x: F.relu(x)

        if self._args.deep:
            card_o = A(self.cards_fc_1(card_o))
            card_o = A(self.cards_fc_2(card_o) + card_o)
            card_o = A(self.cards_fc_3(card_o))

            hist_o = A(self.history_1(hist_o))
            hist_o = A(self.history_2(hist_o) + hist_o)

            # ct_o = A(self.option_1(ct_o))
            # ct_o = A(self.option_2(ct_o) + ct_o)

            y = A(self.comb_1(torch.cat([card_o, hist_o], dim=-1)))
            y = A(self.comb_2(y) + y)

        else:
            y = torch.cat([hist_o, card_o], dim=-1)
            y = A(self.layer_1(y))
            y = A(self.layer_2(y) + y)
            y = A(self.layer_3(y) + y)

        # """""""""""""""""""""""
        # Normalize last layer
        # """""""""""""""""""""""
        if self._args.normalize:
            y = self.norm(y)

        return y


class LowMPMArgs:

    def __init__(self,
                 deep=True,
                 dim=128,
                 dropout=0.0,
                 normalize=True,
                 dim_c=3
                 ):
        self.deep = deep
        self.dim = dim
        self.dropout = dropout
        self.normalize = normalize
        self.dim_c = dim_c

    def get_mpm_cls(self):
        return LowMainPokerModule
