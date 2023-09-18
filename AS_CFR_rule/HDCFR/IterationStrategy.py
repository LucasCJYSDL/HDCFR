import numpy as np
import torch
from torch.nn import functional as F

from PokerRL.rl import rl_util
from HDCFR.workers.la.neural.HighAdvNet import HighAdvVet
from HDCFR.workers.la.neural.LowAdvNet import LowAdvNet

class IterationStrategy:

    def __init__(self, t_prof, owner, env_bldr, device, cfr_iter):
        self._t_prof = t_prof
        self._owner = owner
        self._env_bldr = env_bldr
        self._device = device
        self._iteration = cfr_iter

        self._high_adv_net = None
        self._low_adv_net = None
        self._all_range_idxs = torch.arange(self._env_bldr.rules.RANGE_SIZE, device=self._device, dtype=torch.long)

    @property
    def owner(self):
        return self._owner

    @property
    def iteration(self):
        return self._iteration

    @property
    def device(self):
        return self._device

    def reset(self):
        self._high_adv_net = None
        self._low_adv_net = None

    def get_a_probs(self, pub_obses, range_idxs, option_idxs, legal_actions_lists, to_np=True):

        with torch.no_grad():
            legal_action_masks = rl_util.batch_get_legal_action_mask_torch(n_actions=self._env_bldr.N_ACTIONS,
                                                              legal_actions_lists=legal_actions_lists,
                                                              device=self._device, dtype=torch.float32)

            bs = len(range_idxs)
            if self._iteration == 0:  # at iteration 0
                uniform_even_legal = legal_action_masks / (legal_action_masks.sum(-1)
                                                           .unsqueeze(-1)
                                                           .expand_as(legal_action_masks))
                if to_np:
                    return uniform_even_legal.cpu().numpy()
                return uniform_even_legal

            else:
                range_idxs = torch.tensor(range_idxs, dtype=torch.long, device=self._device)
                option_idxs = torch.tensor(option_idxs, dtype=torch.long, device=self._device)

                advantages = self._low_adv_net(pub_obses=pub_obses,
                                               range_idxs=range_idxs,
                                               option_idxs=option_idxs,
                                               legal_action_masks=legal_action_masks)

                # """"""""""""""""""""
                relu_advantages = F.relu(advantages, inplace=False)  # Cause the sum of *positive* regret matters in CFR
                sum_pos_adv_expanded = relu_advantages.sum(1).unsqueeze(-1).expand_as(relu_advantages)

                # """"""""""""""""""""
                # In case all negative
                # """"""""""""""""""""
                # best_legal_deterministic = torch.zeros((bs, self._env_bldr.N_ACTIONS,), dtype=torch.float32,
                #                                        device=self._device)
                # bests = torch.argmax(
                #     torch.where(legal_action_masks.byte(), advantages, torch.full_like(advantages, fill_value=-10e20))
                #     , dim=1
                # )
                # print("1: ", best_legal_deterministic.shape, torch.where(legal_action_masks.byte(), advantages,
                # torch.full_like(advantages, fill_value=-10e20)).shape) # (bs, dim_a)
                # print("2: ", torch.where(legal_action_masks.byte(), advantages, torch.full_like(advantages, fill_value=-10e20)))
                # print("3: ", bests, bests.shape) # (bs, )
                # _batch_arranged = torch.arange(bs, device=self._device, dtype=torch.long)
                # best_legal_deterministic[_batch_arranged, bests] = 1
                # print("4: ", best_legal_deterministic, best_legal_deterministic.shape) # (bs, dim_a)

                # """"""""""""""""""""
                # Strat
                # """"""""""""""""""""
                # they use greedy strategy for all-negative cases rather than uniformly random policy
                # strategy = torch.where(
                #     sum_pos_adv_expanded > 0,
                #     relu_advantages / sum_pos_adv_expanded,
                #     best_legal_deterministic
                # )
                # print("5: ", strategy) # (bs, dim_a)

                # if we stick to the paper and use regret match
                uniform_even_legal = legal_action_masks / (legal_action_masks.sum(-1)
                                                           .unsqueeze(-1)
                                                           .expand_as(legal_action_masks)) # (bs, dim_a)
                strategy = torch.where(
                    sum_pos_adv_expanded > 0,
                    relu_advantages / sum_pos_adv_expanded,
                    uniform_even_legal
                )

                if to_np:
                    strategy = strategy.cpu().numpy()
                return strategy


    def get_action(self, pub_obses, range_idxs, option_idxs, legal_actions_lists):

        a_probs = self.get_a_probs(pub_obses, range_idxs, option_idxs, legal_actions_lists, to_np=False)
        return torch.multinomial(a_probs, num_samples=1).cpu().numpy() # (bs, 1)

    def get_z_probs(self, pub_obses, range_idxs, option_idxs, to_np=True):
        with torch.no_grad():

            option_dim = self._high_adv_net.get_option_dim()
            bs = len(range_idxs)
            legal_options = torch.ones((bs, option_dim), dtype=torch.float32, device=self._device)

            if self._iteration == 0:  # at iteration 0
                uniform_even_legal = legal_options / (legal_options.sum(-1).unsqueeze(-1).expand_as(legal_options))
                if to_np:
                    return uniform_even_legal.cpu().numpy()
                return uniform_even_legal

            else:
                range_idxs = torch.tensor(range_idxs, dtype=torch.long, device=self._device)
                option_idxs = torch.tensor(option_idxs, dtype=torch.long, device=self._device)

                advantages = self._high_adv_net(pub_obses=pub_obses,
                                               range_idxs=range_idxs,
                                               option_idxs=option_idxs)

                # """"""""""""""""""""
                relu_advantages = F.relu(advantages, inplace=False)  # Cause the sum of *positive* regret matters in CFR
                sum_pos_adv_expanded = relu_advantages.sum(1).unsqueeze(-1).expand_as(relu_advantages)

                # """"""""""""""""""""
                # In case all negative
                # """"""""""""""""""""
                # best_legal_deterministic = torch.zeros((bs, option_dim), dtype=torch.float32, device=self._device)
                # bests = torch.argmax(advantages, dim=1)
                # _batch_arranged = torch.arange(bs, device=self._device, dtype=torch.long)
                # best_legal_deterministic[_batch_arranged, bests] = 1
                # print("4: ", best_legal_deterministic, best_legal_deterministic.shape) # (bs, dim_a)

                # """"""""""""""""""""
                # Strat
                # """"""""""""""""""""
                # they use greedy strategy for all-negative cases rather than uniformly random policy
                # strategy = torch.where(
                #     sum_pos_adv_expanded > 0,
                #     relu_advantages / sum_pos_adv_expanded,
                #     best_legal_deterministic
                # )
                # print("5: ", strategy) # (bs, dim_a)

                # if we stick to the paper and use regret match
                uniform_even_legal = legal_options / (legal_options.sum(-1).unsqueeze(-1).expand_as(legal_options)) # (bs, dim_a)
                strategy = torch.where(
                    sum_pos_adv_expanded > 0,
                    relu_advantages / sum_pos_adv_expanded,
                    uniform_even_legal
                )

                if to_np:
                    strategy = strategy.cpu().numpy()
                return strategy

    def get_option(self, pub_obses, range_idxs, option_idxs):

        z_probs = self.get_z_probs(pub_obses, range_idxs, option_idxs, to_np=False)
        return torch.multinomial(z_probs, num_samples=1).cpu().numpy()

    def state_dict(self):
        return {
            "owner": self._owner,
            "net": self.net_state_dict(),
            "iter": self._iteration,
        }

    def net_state_dict(self):
        """ This just wraps the net.state_dict() with the option of returning None if net is None """
        if self._high_adv_net is None:
            high_state_dict = None
        else:
            high_state_dict = self._high_adv_net.state_dict()
        if self._low_adv_net is None:
            low_state_dict = None
        else:
            low_state_dict = self._low_adv_net.state_dict()

        return (high_state_dict, low_state_dict)

    @staticmethod
    def build_from_state_dict(t_prof, env_bldr, device, state):
        s = IterationStrategy(t_prof=t_prof, env_bldr=env_bldr, device=device,
                              owner=state["owner"], cfr_iter=state["iter"])
        s.load_state_dict(state=state)  # loads net state
        return s

    def load_state_dict(self, state):
        assert self._owner == state["owner"]
        assert self._iteration == state["iter"]
        self.load_net_state_dict(state["net"])

    def load_net_state_dict(self, state_dict):
        # if state_dict[0]:
        self._high_adv_net = HighAdvVet(env_bldr=self._env_bldr, device=self._device,
                                        args=self._t_prof.module_args["adv_training"].high_adv_net_args)
        self._high_adv_net.load_state_dict(state_dict[0])
        self._high_adv_net.eval()
        for param in self._high_adv_net.parameters():
            param.requires_grad = False

        # if state_dict[1]:
        self._low_adv_net = LowAdvNet(env_bldr=self._env_bldr, device=self._device,
                                      args=self._t_prof.module_args["adv_training"].low_adv_net_args)
        self._low_adv_net.set_option_emb(self._high_adv_net.get_option_emb())
        self._low_adv_net.load_state_dict(state_dict[1])
        self._low_adv_net.eval()
        for param in self._low_adv_net.parameters():
            param.requires_grad = False

    def get_copy(self, device=None):
        _device = self._device if device is None else device
        return IterationStrategy.build_from_state_dict(t_prof=self._t_prof, env_bldr=self._env_bldr,
                                                       device=_device, state=self.state_dict())
