import numpy as np
import datetime

from DREAM_and_DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DREAM_and_DeepCFR.TrainingProfile import TrainingProfile
from DREAM_and_DeepCFR.workers.driver.Driver import Driver
from HYPERS import *
from PokerRL.game.games import StandardLeduc, BigLeduc, LargeLeduc  # or any other game

if __name__ == '__main__':

    t = datetime.datetime.now()
    t_str = t.strftime('%Y-%m-%d-%H-%M-%S')
    # name_str = "Leduc_OSSDCFR"
    # name_str = "Big_Leduc_OSSDCFR"
    name_str="Large_Leduc_OSSDCFR"
    # single mode is by default
    ctrl = Driver(t_prof=TrainingProfile(
        checkpoint_freq=500,
        log_verbose=False,
        name=name_str,
        path_data='./saved_data/' + name_str + '_' + t_str,
        n_batches_adv_training=SDCFR_LEDUC_BATCHES,
        n_traversals_per_iter=SDCFR_LEDUC_TRAVERSALS_OS,

        eval_modes_of_algo=(
            EvalAgentDeepCFR.EVAL_MODE_SINGLE,
        ),

        sampler="mo",
        n_actions_traverser_samples=1,

        os_eps=OS_EPS,

        # game_cls=StandardLeduc,
        # game_cls=BigLeduc,
        game_cls=LargeLeduc,
        periodic_restart=5,

        DISTRIBUTED=False,

        device_inference="cuda:2",
        device_training="cuda:2",
        device_parameter_server="cpu",
    ),
        eval_methods={
            "br": 50,
        },
        n_iterations=1000
    )
    ctrl.run()
