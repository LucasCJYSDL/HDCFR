import numpy as np
import datetime

from DREAM_and_DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DREAM_and_DeepCFR.TrainingProfile import TrainingProfile
from DREAM_and_DeepCFR.workers.driver.Driver import Driver
from HYPERS import *
from PokerRL.game.games import StandardLeduc, BigLeduc, LargeLeduc_10, LargeLeduc_15, LargeLeduc_20  # or any other game

if __name__ == '__main__':
    t = datetime.datetime.now()
    t_str = t.strftime('%Y-%m-%d-%H-%M-%S')
    # name_str = "Leduc_DREAM"
    # name_str = "Big_Leduc_Dream"
    # name_str = "Large_Leduc_DREAM_10"
    # name_str = "Large_Leduc_DREAM_15"
    name_str = "Large_Leduc_DREAM_20"

    is_distributed = False
    if not is_distributed:
        N_LA_LEDUC_CFR = 1

    ctrl = Driver(t_prof=TrainingProfile(
        checkpoint_freq=50,
        log_verbose=False,
        name=name_str,
        path_data='./saved_data/' + name_str + '_' + t_str,
        n_learner_actor_workers=N_LA_LEDUC_CFR,
        nn_type="feedforward",
        n_batches_adv_training=SDCFR_LEDUC_BATCHES,
        sampler="learned_baseline",
        n_batches_per_iter_baseline=SDCFR_LEDUC_BASELINE_BATCHES,

        n_traversals_per_iter=SDCFR_LEDUC_TRAVERSALS_OS / N_LA_LEDUC_CFR * 2, # *2
        batch_size_baseline=int(512 / N_LA_LEDUC_CFR),
        mini_batch_size_adv=int(2048 / N_LA_LEDUC_CFR),
        max_buffer_size_adv=int(2e6 / N_LA_LEDUC_CFR),

        os_eps=OS_EPS,
        # game_cls=StandardLeduc,
        # game_cls=BigLeduc,
        # game_cls=LargeLeduc_10,
        # game_cls=LargeLeduc_15,
        game_cls=LargeLeduc_20,
        periodic_restart=5, # danger

        eval_modes_of_algo=(
            EvalAgentDeepCFR.EVAL_MODE_AVRG_NET,
        ), # time-consuming to do both

        n_batches_avrg_training=4000, # time-consuming

        DISTRIBUTED=is_distributed,

        device_inference="cpu",
        device_training="cpu",
        device_parameter_server="cpu",

    ),
        eval_methods={
            "br": 50,
        },
        n_iterations=610,
    )
    ctrl.run()
