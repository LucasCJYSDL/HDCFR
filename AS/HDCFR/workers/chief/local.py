import copy
from os.path import join as ospj

from PokerRL.rl import rl_util
from PokerRL.rl.base_cls.workers.ChiefBase import ChiefBase as _ChiefBase
from PokerRL.util import file_util

from HDCFR.EvalAgent import EvalAgent


class Chief(_ChiefBase):
    # interact with the eval agent: pull strategy from ps and send it to eval_agent
    def __init__(self, t_prof):
        super().__init__(t_prof=t_prof)
        self._ps_handles = None
        self._la_handles = None
        self._env_bldr = rl_util.get_env_builder(t_prof=t_prof)

        self._SINGLE = EvalAgent.EVAL_MODE_SINGLE in self._t_prof.eval_modes_of_algo
        self._AVRG = EvalAgent.EVAL_MODE_AVRG_NET in self._t_prof.eval_modes_of_algo

        assert self._AVRG and not self._SINGLE, "The single mode is not part of our algorthm design."

    def get_log_buf(self):
        return self._log_buf

    def set_la_handles(self, *la_handles):
        self._la_handles = list(la_handles)

    def set_ps_handle(self, *ps_handles):
        self._ps_handles = list(ps_handles)

    def update_alive_las(self, alive_la_handles):
        self._la_handles = alive_la_handles

    # ____________________________________________________ Strategy ____________________________________________________
    def pull_current_eval_strategy(self, receiver_name):
        d = {}
        d[EvalAgent.EVAL_MODE_AVRG_NET] = self._pull_avrg_net_eval_strat() # (high_net, low_net)
        return d

    def _pull_avrg_net_eval_strat(self):
        return [
            self._ray.get(self._ray.remote(ps.get_avrg_weights))
            for ps in self._ps_handles
        ] # clearly, get_avrg_weights needs to be modified


    def export_agent(self, step):
        _dir = ospj(self._t_prof.path_agent_export_storage, str(self._t_prof.name), str(step))
        file_util.create_dir_if_not_exist(_dir)

        MODE = EvalAgent.EVAL_MODE_AVRG_NET

        t_prof = copy.deepcopy(self._t_prof)
        t_prof.eval_modes_of_algo = [MODE]

        eval_agent = EvalAgent(t_prof=t_prof)
        eval_agent.reset()

        w = {EvalAgent.EVAL_MODE_AVRG_NET: self._pull_avrg_net_eval_strat()}
        eval_agent.update_weights(w)
        eval_agent.set_mode(mode=MODE)
        eval_agent.store_to_disk(path=_dir, file_name="eval_agent" + MODE)

