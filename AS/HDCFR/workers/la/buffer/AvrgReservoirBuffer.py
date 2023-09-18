import torch

from HDCFR.workers.la.buffer._ReservoirBufferBase import ReservoirBufferBase as _ResBufBase
from PokerRL.rl import rl_util


class AvrgReservoirBuffer(_ResBufBase):
    """
    Reservoir buffer to store state+action samples for the average strategy network
    """

    def __init__(self, owner, nn_type, max_size, env_bldr, iter_weighting_exponent, t_prof):
        super().__init__(owner=owner, max_size=max_size, env_bldr=env_bldr, nn_type=nn_type,
                         iter_weighting_exponent=iter_weighting_exponent)

        self._z_probs_buffer = torch.zeros((max_size, t_prof.dim_c), dtype=torch.float32, device=self.device)
        self._a_probs_buffer = torch.zeros((max_size, env_bldr.N_ACTIONS), dtype=torch.float32, device=self.device)

    def state_dict(self):
        return {
            "base": super().state_dict(),
            "z_probs": self._z_probs_buffer,
            "a_probs": self._a_probs_buffer
        }

    def load_state_dict(self, state):
        super().load_state_dict(state["base"])
        self._z_probs_buffer = state["z_probs"]
        self._a_probs_buffer = state["a_probs"]

    def sample(self, batch_size, device):
        indices = torch.randint(0, self.size, (batch_size,), dtype=torch.long, device=self.device)

        if self._nn_type == "recurrent":
            obses = self._pub_obs_buffer[indices.cpu().numpy()]
        elif self._nn_type == "feedforward":
            obses = self._pub_obs_buffer[indices].to(device)
        else:
            raise NotImplementedError

        return \
            obses, \
            self._range_idx_buffer[indices].to(device), \
            self._last_option_buffer[indices].to(device), \
            self._cur_option_buffer[indices].to(device), \
            self._legal_action_mask_buffer[indices].to(device), \
            self._z_probs_buffer[indices].to(device), \
            self._a_probs_buffer[indices].to(device), \
            self._high_iteration_buffer[indices].to(device) / self._last_high_iteration_seen, \
            self._low_iteration_buffer[indices].to(device) / self._last_low_iteration_seen

    def add(self, pub_obs, range_idx, last_option, cur_option, legal_actions_list, z_probs, a_probs, high_iteration, low_iteration):
        if self.size < self._max_size:
            self._add(idx=self.size,
                      pub_obs=pub_obs,
                      range_idx=range_idx,
                      last_option=last_option,
                      cur_option=cur_option,
                      legal_action_mask=self._get_mask(legal_actions_list),
                      option_probs=z_probs,
                      action_probs=a_probs,
                      high_iteration=high_iteration,
                      low_iteration=low_iteration)
            self.size += 1

        elif self._should_add():
            self._add(idx=self._random_idx(),
                      pub_obs=pub_obs,
                      range_idx=range_idx,
                      last_option=last_option,
                      cur_option=cur_option,
                      legal_action_mask=self._get_mask(legal_actions_list),
                      option_probs=z_probs,
                      action_probs=a_probs,
                      high_iteration=high_iteration,
                      low_iteration=low_iteration)

        self.n_entries_seen += 1

    def _get_mask(self, legal_actions_list):
        return rl_util.get_legal_action_mask_torch(n_actions=self._env_bldr.N_ACTIONS,
                                                   legal_actions_list=legal_actions_list,
                                                   device=self.device, dtype=torch.float32)

    def _add(self, idx, pub_obs, range_idx, last_option, cur_option, legal_action_mask, option_probs, action_probs, high_iteration, low_iteration):
        if self._nn_type == "feedforward":
            pub_obs = torch.from_numpy(pub_obs)

        self._pub_obs_buffer[idx] = pub_obs
        self._range_idx_buffer[idx] = range_idx
        self._last_option_buffer[idx] = last_option
        self._cur_option_buffer[idx] = cur_option
        self._legal_action_mask_buffer[idx] = legal_action_mask
        self._z_probs_buffer[idx] = option_probs
        self._a_probs_buffer[idx] = action_probs

        # In "https://arxiv.org/pdf/1811.00164.pdf", Brown et al. weight by floor((t+1)/2), but we assume that
        # this is due to incrementation happening for every alternating update. We count one iteration as an
        # update for both players.
        self._high_iteration_buffer[idx] = float(high_iteration) ** self._iter_weighting_exponent
        self._last_high_iteration_seen = high_iteration

        self._low_iteration_buffer[idx] = float(low_iteration) ** self._iter_weighting_exponent
        self._last_low_iteration_seen = low_iteration