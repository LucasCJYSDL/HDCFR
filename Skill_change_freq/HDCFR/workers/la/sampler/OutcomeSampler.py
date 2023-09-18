import numpy as np
import torch
from typing import List
from tqdm import tqdm

from PokerRL.rl import rl_util
from PokerRL.game.PokerEnvStateDictEnums import EnvDictIdxs

from HDCFR.IterationStrategy import IterationStrategy
from HDCFR.workers.la.wrapper.BaselineWrapper import BaselineWrapper
from HDCFR.workers.la.buffer.AdvReservoirBuffer import AdvReservoirBuffer
from HDCFR.workers.la.buffer.AvrgReservoirBuffer import AvrgReservoirBuffer
from HDCFR.workers.la.buffer.BaselineBuffer import BaselineBuffer

class OutcomeSampler:
    def __init__(self, env_bldr, adv_buffers: List[AdvReservoirBuffer], baseline_net: BaselineWrapper, baseline_buf: BaselineBuffer,
                 t_prof, avrg_buffers: List[AvrgReservoirBuffer], eps=0.5):
        self._env_bldr = env_bldr
        self._adv_buffers = adv_buffers
        self._avrg_buffers = avrg_buffers
        self._env_wrapper = self._env_bldr.get_new_wrapper(is_evaluating=False)

        self._baseline_net = baseline_net
        self._baseline_buf = baseline_buf

        self._eps = eps
        self._actions_arranged = np.arange(self._env_bldr.N_ACTIONS)
        self.total_node_count_traversed = 0

        self.option_dim = t_prof.dim_c

    def generate(self, n_traversals, traverser, iteration_strats: List[IterationStrategy], cfr_iter):
        for _ in range(n_traversals):
            self._traverse_once(traverser=traverser, iteration_strats=iteration_strats, cfr_iter=cfr_iter)

    def _traverse_once(self, traverser, iteration_strats: List[IterationStrategy], cfr_iter):

        self._env_wrapper.reset()
        self._recursive_high_traversal(start_state_dict=self._env_wrapper.state_dict(),
                                       traverser=traverser,
                                       trav_depth=0,
                                       plyrs_range_idxs=[
                                           self._env_wrapper.env.get_range_idx(p_id=p_id)
                                           for p_id in range(self._env_bldr.N_SEATS)
                                       ],
                                       sample_reach=1.0,
                                       non_traverser_reach=1.0,
                                       iteration_strats=iteration_strats,
                                       cfr_iter=cfr_iter,
                                       last_options=[self.option_dim, self.option_dim],
                                       traj_data={"pub_obs": [], "range_idx": [], "range_idx_list": [], "cur_option": [],
                                                  "legal_action": [], "legal_action_mask": [], "action": [],
                                                  "player": [], "q_h_z": [], "q_h_z_a": [], "last_option": [],
                                                  "b_h_z": [], "b_h_z_a": []})

    def _recursive_high_traversal(self, start_state_dict, traverser, trav_depth, plyrs_range_idxs, last_options,
                                  iteration_strats: List[IterationStrategy], cfr_iter, sample_reach, non_traverser_reach,
                                  traj_data):
        self.total_node_count_traversed += 1 # only add up for h rather than hz
        self._env_wrapper.load_state_dict(start_state_dict)

        cur_player = start_state_dict["base"]["env"][EnvDictIdxs.current_player]
        assert cur_player == self._env_wrapper.env.current_player.seat_id

        pub_obs_t = self._env_wrapper.get_current_obs()
        player_range_idx = plyrs_range_idxs[cur_player]
        z_prob = iteration_strats[cur_player].get_z_probs(pub_obses=[pub_obs_t], range_idxs=[player_range_idx],
                                                          option_idxs=[last_options[cur_player]], to_np=False)[0].cpu() # (option_dim, )

        if cur_player == traverser:
            legal_options = torch.ones((self.option_dim, ), dtype=torch.float32, device=self._adv_buffers[cur_player].device)
            sample_strat = (1 - self._eps) * z_prob + self._eps * (legal_options.cpu() / self.option_dim)
            # sample_strat = legal_options.cpu() / float(self.option_dim)
        else:
            sample_strat = z_prob

        cur_option = torch.multinomial(sample_strat, num_samples=1).item() # scalar
        cur_options = last_options.copy()
        cur_options[cur_player] = cur_option
        # print("3: ", last_options, cur_options)

        if cur_player == traverser:
            sample_reach_sup = sample_strat[cur_option].item() * self.option_dim
        else:
            sample_reach_sup = 1.0

        # making use of the baseline function
        legal_actions_list = self._env_wrapper.env.get_legal_actions()
        legal_action_mask = rl_util.get_legal_action_mask_torch(n_actions=self._env_bldr.N_ACTIONS,
                                                                legal_actions_list=legal_actions_list,
                                                                device=self._adv_buffers[cur_player].device,
                                                                dtype=torch.float32)

        baseline_range_id = plyrs_range_idxs[0] * 10000 + plyrs_range_idxs[1]
        b_list = self._baseline_net.get_b(pub_obses=[pub_obs_t for _ in range(self.option_dim)],
                                          range_idxs=[baseline_range_id for _ in range(self.option_dim)],
                                          option_idxs=[i for i in range(self.option_dim)],
                                          legal_actions_lists=[legal_actions_list for _ in range(self.option_dim)],
                                          to_np=True)  # (option_dim, action_dim)
        a_prob_list = iteration_strats[cur_player].get_a_probs(
            pub_obses=[pub_obs_t for _ in range(self.option_dim)],
            range_idxs=[player_range_idx for _ in range(self.option_dim)],
            option_idxs=[i for i in range(self.option_dim)],
            legal_actions_lists=[legal_actions_list for _ in range(self.option_dim)],
            to_np=True)  # (option_dim, action_dim)

        z_a_prob = z_prob.unsqueeze(1).repeat(1, a_prob_list.shape[-1]) * torch.tensor(a_prob_list)

        b_h_z = (b_list * a_prob_list).sum(axis=1)  # (option_dim, )
        # print("1: ", b_list, a_prob_list, b_h_z)
        traj_data["last_option"].append(last_options[cur_player])
        traj_data["q_h_z"].append(sample_strat[cur_option].item())
        traj_data["b_h_z"].append(b_h_z[cur_option])

        if cur_player != traverser:
            new_non_traverser_reach = non_traverser_reach * sample_strat[cur_option].item()
        else:
            new_non_traverser_reach = non_traverser_reach * 1.0

        v_hz, low_regrets, a_prob = self._recursive_low_traversal(
            start_state_dict=self._env_wrapper.state_dict(), # may be not necessary
            traverser=traverser,
            trav_depth=trav_depth, # only add it up by 1 after the env.step
            plyrs_range_idxs=plyrs_range_idxs,
            iteration_strats=iteration_strats,
            cfr_iter=cfr_iter,
            sample_reach=sample_reach * sample_reach_sup,
            non_traverser_reach=new_non_traverser_reach,
            cur_options=cur_options,
            traj_data=traj_data
        )

        v_h_z = b_h_z.copy()
        v_h_z[cur_option] += (v_hz - b_h_z[cur_option]) / sample_strat[cur_option].item() # TODO: denominator cannot be 0

        v_h = (z_prob.numpy() * v_h_z).sum()
        # print("2: ", z_prob.numpy(), v_h_z, v_h)
        # update the buffers
        if cur_player == traverser: # TODO: add adv, no matter whether cur_player == traverser
            # TODO: denominator cannot be 0 # (option_dim, )
            # high_regrets = (v_h_z - v_h) * (1.0 if traverser == 0 else -1.0) * non_traverser_reach / sample_reach
            high_regrets = (v_h_z - v_h) * (1.0 if traverser == 0 else -1.0)
            self._adv_buffers[cur_player].add(pub_obs=pub_obs_t,
                                              range_idx=player_range_idx,
                                              last_option=last_options[cur_player],
                                              cur_option=cur_option,
                                              legal_action_mask=legal_action_mask,
                                              high_adv=high_regrets,
                                              low_adv=low_regrets,
                                              high_iteration=(cfr_iter + 1) / sample_reach,
                                              low_iteration=(cfr_iter + 1) / sample_reach / sample_reach_sup) # TODO: try (cfr_iter+1)/sample_reach

        else:
            self._avrg_buffers[cur_player].add(pub_obs=pub_obs_t,
                                               range_idx=player_range_idx,
                                               last_option=last_options[cur_player],
                                               cur_option=cur_option,
                                               legal_actions_list=legal_actions_list,
                                               z_probs=z_prob.to(device=self._avrg_buffers[cur_player].device).detach().clone(),
                                               a_probs=a_prob.to(device=self._avrg_buffers[cur_player].device).detach().clone(),
                                               high_iteration=(cfr_iter + 1) / sample_reach,
                                               low_iteration=(cfr_iter + 1) / sample_reach / sample_reach_sup) # TODO: try (cfr_iter+1)/sample_reach


        return v_h, z_a_prob

    def _recursive_low_traversal(self, start_state_dict, traverser, trav_depth, plyrs_range_idxs, iteration_strats,
                                 cfr_iter, sample_reach, non_traverser_reach, cur_options, traj_data):
        self._env_wrapper.load_state_dict(start_state_dict) # not necessary, TODO: test

        cur_player = start_state_dict["base"]["env"][EnvDictIdxs.current_player]
        assert cur_player == self._env_wrapper.env.current_player.seat_id

        pub_obs_t = self._env_wrapper.get_current_obs()
        player_range_idx = plyrs_range_idxs[cur_player]
        legal_actions_list = self._env_wrapper.env.get_legal_actions()
        legal_action_mask = rl_util.get_legal_action_mask_torch(n_actions=self._env_bldr.N_ACTIONS,
                                                                legal_actions_list=legal_actions_list,
                                                                device=self._adv_buffers[cur_player].device,
                                                                dtype=torch.float32)

        a_prob = iteration_strats[cur_player].get_a_probs(
            pub_obses=[pub_obs_t],
            range_idxs=[player_range_idx],
            option_idxs=[cur_options[cur_player]],
            legal_actions_lists=[legal_actions_list],
            to_np=False
        )[0].cpu() # (action_dim, )

        if cur_player == traverser:
            n_legal_actions = len(legal_actions_list)
            sample_strat = (1 - self._eps) * a_prob + self._eps * (legal_action_mask.cpu() / n_legal_actions)
            # sample_strat = legal_action_mask.cpu() / float(n_legal_actions)
        else:
            sample_strat = a_prob

        cur_action = torch.multinomial(sample_strat, num_samples=1).item()
        pub_obs_tp1, rew_for_all, done, _info = self._env_wrapper.step(cur_action)
        legal_action_mask_tp1 = rl_util.get_legal_action_mask_torch(n_actions=self._env_bldr.N_ACTIONS,
                                                                    legal_actions_list=self._env_wrapper.env.get_legal_actions(),
                                                                    device=self._adv_buffers[traverser].device,
                                                                    dtype=torch.float32)

        if cur_player == traverser:
            sample_reach_sup = sample_strat[cur_action].item() * n_legal_actions
        else:
            sample_reach_sup = 1.0

        # making use of the baseline function
        baseline_range_id = plyrs_range_idxs[0] * 10000 + plyrs_range_idxs[1]
        b_h_z_a = self._baseline_net.get_b(
            pub_obses=[pub_obs_t],
            range_idxs=[baseline_range_id],
            option_idxs=[cur_options[cur_player]],
            legal_actions_lists=[legal_actions_list],
            to_np=True
        )[0]  # (action_dim, )

        traj_data["pub_obs"].append(pub_obs_t)
        traj_data["range_idx"].append(baseline_range_id)
        traj_data["range_idx_list"].append(player_range_idx)
        traj_data["cur_option"].append(cur_options[cur_player])
        traj_data["action"].append(cur_action)
        traj_data["legal_action_mask"].append(legal_action_mask.cpu().numpy())
        traj_data["player"].append(cur_player)
        traj_data["legal_action"].append(legal_actions_list)
        traj_data["q_h_z_a"].append(sample_strat[cur_action].item())
        traj_data["b_h_z_a"].append(b_h_z_a[cur_action])

        if done:
            v_hza = rew_for_all[0] # the reward of player 0 is used as the basement
            self.total_node_count_traversed += 1
            z_a_prob = torch.zeros_like(a_prob)
        else:
            if cur_player != traverser:
                new_non_traverser_reach = non_traverser_reach * sample_strat[cur_action].item()
            else:
                new_non_traverser_reach = non_traverser_reach * 1.0

            v_hza, z_a_prob = self._recursive_high_traversal(
                start_state_dict=self._env_wrapper.state_dict(),
                traverser=traverser,
                trav_depth=trav_depth+1,  # only add it up by 1 after the env.step
                plyrs_range_idxs=plyrs_range_idxs,
                iteration_strats=iteration_strats,
                cfr_iter=cfr_iter,
                sample_reach=sample_reach * sample_reach_sup,
                non_traverser_reach=new_non_traverser_reach,
                last_options=cur_options,
                traj_data=traj_data
            )

        v_hz_a = b_h_z_a.copy()
        v_hz_a[cur_action] += (v_hza - b_h_z_a[cur_action]) / sample_strat[cur_action].item()
        v_hz = (a_prob.numpy() * v_hz_a).sum()

        # low_regrets = (v_hz_a - v_hz) * (1 if cur_player == 0 else -1) * non_traverser_reach / sample_reach
        low_regrets = (v_hz_a - v_hz) * (1 if cur_player == 0 else -1) * legal_action_mask.cpu().numpy() # the sample reach part is included in the cfr iteration

        self._baseline_buf.add(
            pub_obs=pub_obs_t,
            range_idx_crazy_embedded=baseline_range_id,
            cur_option=cur_options[cur_player],
            legal_action_mask=legal_action_mask,
            r=rew_for_all[0],
            a=cur_action,
            done=done,
            pub_obs_tp1=pub_obs_tp1,
            strat_tp1=z_a_prob,
            legal_action_mask_tp1=legal_action_mask_tp1,
        )

        return v_hz, low_regrets, a_prob




