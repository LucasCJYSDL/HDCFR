import numpy as np

from HDCFR.workers.la.wrapper.AvrgWrapper import HierAvrgWrapper
from HDCFR.EvalAgentBase import EvalAgentBase as _EvalAgentBase

class EvalAgent(_EvalAgentBase):
    EVAL_MODE_AVRG_NET = "AVRG_NET"
    EVAL_MODE_SINGLE = "SINGLE"
    ALL_MODES = [EVAL_MODE_AVRG_NET, EVAL_MODE_SINGLE]

    def __init__(self, t_prof, mode=None, device=None):
        super().__init__(t_prof=t_prof, mode=mode, device=device) # ray, device, env_builder, env_wrapper (gym-like stuff)
        self.avrg_args = t_prof.module_args["avrg_training"]

        self._AVRG = EvalAgent.EVAL_MODE_AVRG_NET in self.t_prof.eval_modes_of_algo
        self._SINGLE = EvalAgent.EVAL_MODE_SINGLE in self.t_prof.eval_modes_of_algo

        assert self._AVRG and not self._SINGLE, "The single mode is not part of our algorthm design."

        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        self.avrg_net_policies = [
            HierAvrgWrapper(avrg_training_args=self.avrg_args, owner=p, env_bldr=self.env_bldr, device=self.device)
            for p in range(t_prof.n_seats)
        ]
        for pol in self.avrg_net_policies:
            pol.eval()

        self.n_seats = t_prof.n_seats
        self.option_dim = t_prof.dim_c
        self.last_options = None

    def can_compute_mode(self):
        """ All modes are always computable (i.e. not dependent on iteration etc.)"""
        return True

    def get_a_probs_for_each_hand(self, last_option):

        pub_obs = self._internal_env_wrapper.get_current_obs()
        legal_actions_list = self._internal_env_wrapper.env.get_legal_actions()
        p_id_acting = self._internal_env_wrapper.env.current_player.seat_id
        last_option_id = last_option[p_id_acting]

        return self.avrg_net_policies[p_id_acting].get_a_probs_for_each_hand(pub_obs=pub_obs,
                                                                             legal_actions_list=legal_actions_list,
                                                                             option_idx=last_option_id)

    def update_weights(self, weights_for_eval_agent):

        avrg_weights = weights_for_eval_agent[self.EVAL_MODE_AVRG_NET]

        for p in range(self.t_prof.n_seats):
            self.avrg_net_policies[p].load_net_state_dict(
                                                          (self.ray.state_dict_to_torch(avrg_weights[p][0], device=self.device),\
                                                           self.ray.state_dict_to_torch(avrg_weights[p][1], device=self.device))
                                                         )
            self.avrg_net_policies[p].eval()

    def _state_dict(self):
        d = {}
        d["avrg_nets"] = [pol.net_state_dict() for pol in self.avrg_net_policies]
        return d

    def _load_state_dict(self, state):
        for i in range(self.t_prof.n_seats):
            self.avrg_net_policies[i].load_net_state_dict(state["avrg_nets"][i])

    def reset(self, deck_state_dict=None):
        self.last_options = [self.option_dim for _ in range(self.n_seats)]
        super().reset(deck_state_dict=deck_state_dict)

    def get_action(self, step_env=True, need_probs=False):
        assert not need_probs
        p_id_acting = self._internal_env_wrapper.env.current_player.seat_id
        range_idx = self._internal_env_wrapper.env.get_range_idx(p_id=p_id_acting)
        last_option = self.last_options[p_id_acting]

        z_probs = self.avrg_net_policies[p_id_acting].get_z_probs(
            pub_obses=[self._internal_env_wrapper.get_current_obs()],
            range_idxs=np.array([range_idx], dtype=np.int32),
            option_idxs=np.array([last_option], dtype=np.int32)
        )[0]

        cur_option = np.random.choice(np.arange(self.option_dim), p=z_probs)
        self.last_options[p_id_acting] = cur_option
        # print(cur_option)
        a_probs = self.avrg_net_policies[p_id_acting].get_a_probs(
            pub_obses=[self._internal_env_wrapper.get_current_obs()],
            range_idxs=np.array([range_idx], dtype=np.int32),
            option_idxs=np.array([cur_option], dtype=np.int32),
            legal_actions_lists=[self._internal_env_wrapper.env.get_legal_actions()]
        )[0]

        action = np.random.choice(np.arange(self.env_bldr.N_ACTIONS), p=a_probs)
        # print("1: ", self.last_options, cur_option, z_probs, action, a_probs)
        if step_env:
            self._internal_env_wrapper.step(action=action)

        return action, None





