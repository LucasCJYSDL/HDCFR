import numpy as np
import datetime

from DREAM_and_DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DREAM_and_DeepCFR.TrainingProfile import TrainingProfile
from DREAM_and_DeepCFR.workers.driver.Driver import Driver
from HYPERS import *
from PokerRL.game.games import Flop3Holdem, LargeFlop3Holdem  # or any other game

if __name__ == '__main__':
    t = datetime.datetime.now()
    t_str = t.strftime('%Y-%m-%d-%H-%M-%S')
    name_str = "FHP_OSSDCFR"
    # name_str = "LargeFHP_OSSDCFR"

    ctrl = Driver(t_prof=TrainingProfile(
        checkpoint_freq=50,
        name=name_str,
        path_data='./saved_data/' + name_str + '_' + t_str,

        n_traversals_per_iter=(SDCFR_FHP_TRAVERSALS_OS / N_LA_FHP_CFR),
        eval_modes_of_algo=(
            EvalAgentDeepCFR.EVAL_MODE_SINGLE,
        ),
        sampler="mo",
        n_actions_traverser_samples=1,
        os_eps=OS_EPS,

        game_cls=Flop3Holdem,
        # game_cls=LargeFlop3Holdem,
        n_batches_adv_training=SDCFR_FHP_BATCHES,
        n_learner_actor_workers=N_LA_FHP_CFR,
        mini_batch_size_adv=int(SDCFR_FHP_BATCH_SIZE / N_LA_FHP_CFR),
        max_buffer_size_adv=int(4e7 / N_LA_FHP_NFSP),
        DISTRIBUTED=True,
        rlbr_args=DIST_RLBR_ARGS_games,

        periodic_restart=5,
        device_inference="cpu",
        device_training="cpu",
        device_parameter_server="cpu",
    ),
        eval_methods={'h2h_eval': -1},
        n_iterations=1000
    )
    ctrl.run()
