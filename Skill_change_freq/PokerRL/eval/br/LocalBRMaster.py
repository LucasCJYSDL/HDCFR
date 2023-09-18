# Copyright (c) 2019 Eric Steinberger


import copy

from PokerRL.eval._.EvaluatorMasterBase import EvaluatorMasterBase
from PokerRL.game._.tree.PublicTree import PublicTree
from PokerRL.rl import rl_util


class LocalBRMaster(EvaluatorMasterBase):
    """
    (Local) Master to evaluate agents by computing an exact best-response strategy and thereby evaluating the agent's
    exploitability. Note that this type of evaluation should only be used in games with up to a million information sets
    as the provided implementation is not meant for big games. Other evaluators provided are better suited for those.
    """

    def __init__(self, t_prof, chief_handle, eval_agent_cls):
        super().__init__(t_prof=t_prof, eval_env_bldr=rl_util.get_env_builder(t_prof=t_prof), chief_handle=chief_handle,
                         evaluator_name="BR")
        self._env_bldr = rl_util.get_env_builder(t_prof=t_prof)

        self._eval_agent = eval_agent_cls(t_prof=t_prof)

        self._game_trees = [
            PublicTree(env_bldr=self._env_bldr,
                       stack_size=stack_size,
                       stop_at_street=None,
                       put_out_new_round_after_limit=True,
                       is_debugging=self._t_prof.DEBUGGING)
            for stack_size in self._t_prof.eval_stack_sizes
        ]

        for gt in self._game_trees:
            gt.build_tree()
            print("Tree with stack size", gt.stack_size, "has", gt.n_nodes, "nodes out of which", gt.n_nonterm,
                  "are non-terminal.")

    def evaluate(self, iter_nr, is_hier=False):
        # print("here: ", is_hier)
        for mode in self._t_prof.eval_modes_of_algo:

            if self._is_multi_stack:
                total_of_all_stacks = []

            for stack_size_idx, stack_size in enumerate(self._t_prof.eval_stack_sizes):
                # print("2: ", stack_size, stack_size_idx)
                self._eval_agent.set_mode(mode)
                self._eval_agent.set_stack_size(stack_size=stack_size)
                if self._eval_agent.can_compute_mode():
                    expl, ratio = self._compute_br_heads_up(stack_size_idx=stack_size_idx, iter_nr=iter_nr, is_hier=is_hier)
                    self._log_results(iter_nr=iter_nr, agent_mode=mode, stack_size_idx=stack_size_idx, score=expl)

                print("EXPL: ", expl)

                if self._is_multi_stack:
                    total_of_all_stacks.append(expl)

            if self._is_multi_stack:
                self._log_multi_stack(agent_mode=mode,
                                      iter_nr=iter_nr,
                                      score_total=sum(total_of_all_stacks) / float(len(total_of_all_stacks)))

        return ratio

    def update_weights(self):
        w = self.pull_current_strat_from_chief()
        self._eval_agent.update_weights(copy.deepcopy(w)) # the deep copy here is not necessary

    def _compute_br_heads_up(self, stack_size_idx, iter_nr=None, do_export_tree=True, is_hier=False):
        gt = self._game_trees[stack_size_idx]
        ratio = gt.fill_with_agent_policy(agent=self._eval_agent, is_hier=is_hier) # danger
        gt.compute_ev()

        if do_export_tree:
            gt.export_to_file(name=self._t_prof.name + "__BR_vs_" + self._eval_agent.get_mode()
                                   + "__stack_idx"
                                   + str(stack_size_idx) + "_I_" + str(iter_nr))

        # mean over players
        return sum(gt.root.exploitability) * self._env_bldr.env_cls.EV_NORMALIZER / self._t_prof.n_seats, ratio
