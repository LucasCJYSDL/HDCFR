from DREAM_and_DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from HDCFR.EvalAgent import EvalAgent
from NFSP.EvalAgentNFSP import EvalAgentNFSP

from DREAM_and_DeepCFR.TrainingProfile import TrainingProfile as DCFR_TrainingProfile
from HDCFR.TrainingProfile import TrainingProfile as HDCFR_TrainingProfile
from NFSP.TrainingProfile import TrainingProfile as NFSP_TrainingProfile

from DREAM_and_DeepCFR.workers.driver.Driver import Driver as DCFR_Driver
from HDCFR.workers.driver.Driver import Driver as HDCFR_Driver
from NFSP.workers.driver.Driver import Driver as NFSP_Driver

from PokerRL.eval.head_to_head.H2HArgs import H2HArgs
from PokerRL.eval.head_to_head.LocalHead2HeadMaster import LocalHead2HeadMaster

from PokerRL.eval._.EvaluatorMasterBase import EvaluatorMasterBase
from PokerRL.rl import rl_util


def H2H_eval(name_str, t_str, iter, game_cls):

    name_str_ls = [name_str, "FHP_DREAM"]
    t_str_ls = [t_str, "largeeval"]
    iter_ls = [iter, 100] # 600

    ctrl_list = []

    round_num = 1e2 # 1e4

    for player_idx in range(2):
        name_str = name_str_ls[player_idx]
        t_str = t_str_ls[player_idx]
        iter = iter_ls[player_idx]

        alg = name_str.split('_')[1]

        print(alg)

        if alg in ['DREAM', 'SDCFR', 'OSSDCFR']:
            if alg == 'DREAM':
                eval_modes = (EvalAgentDeepCFR.EVAL_MODE_AVRG_NET,)
            else:
                eval_modes = (EvalAgentDeepCFR.EVAL_MODE_SINGLE,)

            sampler = {"DREAM": "learned_baseline", "SDCFR": "es", "OSSDCFR": "mo"}[alg]

            ctrl = DCFR_Driver(t_prof=DCFR_TrainingProfile(
                name=name_str,
                path_data='./saved_data/' + name_str + '_' + t_str,

                sampler=sampler,
                game_cls=game_cls,

                eval_modes_of_algo=eval_modes,

                DISTRIBUTED=False,

                h2h_args=H2HArgs(n_hands=round_num)
            ),
                eval_methods={"h2h": 1},  # danger
                iteration_to_import=iter,
                name_to_import=name_str
            )
            mode_list = [eval_modes[0], eval_modes[0]]

        elif alg == "NFSP":
            ctrl = NFSP_Driver(t_prof=NFSP_TrainingProfile(
                name=name_str,
                path_data='./saved_data/' + name_str + '_' + t_str,
                game_cls=game_cls,
                eval_modes_of_algo=(
                    EvalAgentNFSP.EVAL_MODE_AVG,
                ),
                DISTRIBUTED=False,
                h2h_args=H2HArgs(n_hands=round_num),
            ),
                eval_methods={"h2h": 1},  # danger
                iteration_to_import=iter,
                name_to_import=name_str
            )
            mode_list = [EvalAgentNFSP.EVAL_MODE_AVG, EvalAgentNFSP.EVAL_MODE_AVG]

        else:
            assert alg == "HDCFR"
            ctrl = HDCFR_Driver(
                t_prof=HDCFR_TrainingProfile(
                    name=name_str,
                    dim_c=2,
                    path_data='./saved_data/' + name_str + '_' + t_str,

                    sampler="learned_baseline",
                    game_cls=game_cls,
                    eval_modes_of_algo=(
                        EvalAgent.EVAL_MODE_AVRG_NET,
                    ),

                    DISTRIBUTED=False,
                    h2h_args=H2HArgs(n_hands=round_num),
                ),
                eval_methods={"h2h": 1},
                iteration_to_import=iter,
                name_to_import=name_str
            )
            mode_list = [EvalAgent.EVAL_MODE_AVRG_NET, EvalAgent.EVAL_MODE_AVRG_NET]

        ctrl_list.append(ctrl)

        assert "h2h" in ctrl.eval_masters
        ctrl.eval_masters["h2h"][0].update_weights()
        ctrl.eval_masters["h2h"][0].set_modes(mode_list)

    # a little hack here: we take the first player as the main player and replace its opponent with the second player
    ctrl_0 = ctrl_list[0]
    ctrl_1 = ctrl_list[1]
    eval_agent_0 = ctrl_0.eval_masters["h2h"][0]
    eval_agent_1 = ctrl_1.eval_masters["h2h"][0]

    assert isinstance(eval_agent_0, LocalHead2HeadMaster)
    assert isinstance(eval_agent_1, LocalHead2HeadMaster)

    # eval_agent_1.eval_agents[1] contains two players, so it's still a fair comparison
    eval_agent_0.eval_agents[1] = eval_agent_1.eval_agents[1]
    # evaluate: each agent sits in both seats and plays h_hands
    mean, d = eval_agent_0.evaluate(iter_nr=0)

    return mean, d


class LocalH2HEvalMaster(EvaluatorMasterBase):

    def __init__(self, t_prof, chief_handle, eval_agent_cls):
        super().__init__(t_prof=t_prof, eval_env_bldr=rl_util.get_env_builder(t_prof=t_prof), chief_handle=chief_handle,
                         evaluator_name="Head2Head_Winnings", log_conf_interval=True)
        self.agent_mode = self._t_prof.eval_modes_of_algo[0]

    def update_weights(self):
        pass

    def evaluate(self, eval_iter):
        eval_t_str = self._t_prof._data_path.split('_')[-1]
        mean, d = H2H_eval(name_str=self._t_prof.name, t_str=eval_t_str, iter=eval_iter, game_cls=self._t_prof.game_cls)
        # print("2: ", self._eval_agents[0].get_mode(), self._exp_name_total)
        self._log_results(iter_nr=eval_iter,
                          agent_mode=self.agent_mode,
                          stack_size_idx=0,
                          score=mean, upper_conf95=mean + d, lower_conf95=mean - d)