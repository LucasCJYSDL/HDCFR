import numpy as np
import datetime

from DREAM_and_DeepCFR.TrainingProfile import TrainingProfile
from DREAM_and_DeepCFR.workers.driver.Driver import Driver
from DREAM_and_DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR

from HYPERS import *
from PokerRL import Flop3Holdem, LargeFlop3Holdem

if __name__ == '__main__':
    t = datetime.datetime.now()
    t_str = t.strftime('%Y-%m-%d-%H-%M-%S')
    name_str = "FHP_DREAM"
    # name_str = "LargeFHP_DREAM"

    ctrl = Driver(t_prof=TrainingProfile(
        checkpoint_freq=50,
        name=name_str,
        path_data='./saved_data/' + name_str + '_' + t_str,
        n_traversals_per_iter=(SDCFR_FHP_TRAVERSALS_OS / N_LA_FHP_CFR),
        sampler="learned_baseline",
        os_eps=OS_EPS,

        n_batches_per_iter_baseline=2000,
        batch_size_baseline=int(2048 / N_LA_FHP_CFR),

        eval_modes_of_algo=(
            EvalAgentDeepCFR.EVAL_MODE_AVRG_NET,
        ), # the only difference

        game_cls=Flop3Holdem,
        # game_cls=LargeFlop3Holdem,
        n_batches_adv_training=SDCFR_FHP_BATCHES,
        n_learner_actor_workers=N_LA_FHP_CFR,
        mini_batch_size_adv=int(SDCFR_FHP_BATCH_SIZE / N_LA_FHP_CFR),
        max_buffer_size_adv=int(4e7 / N_LA_FHP_CFR),
        DISTRIBUTED=True,
        rlbr_args=DIST_RLBR_ARGS_games,
        device_inference="cpu",
        device_training="cpu",
        device_parameter_server="cpu",
        periodic_restart=5
    ),
        eval_methods={'h2h_eval': -1},
        n_iterations=1000,
    )
    ctrl.run()
