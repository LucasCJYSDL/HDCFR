import numpy as np
import datetime

from HDCFR.EvalAgent import EvalAgent
from HDCFR.TrainingProfile import TrainingProfile
from HDCFR.workers.driver.Driver import Driver
from HYPERS import *
from PokerRL.game.games import StandardLeduc, BigLeduc, LargeLeduc_10, LargeLeduc_15, LargeLeduc_20  # or any other game

if __name__ == '__main__':
    t = datetime.datetime.now()
    t_str = t.strftime('%Y-%m-%d-%H-%M-%S')
    # name_str = "Leduc_HDCFR"
    # name_str = "Big_Leduc_HDCFR"
    # name_str = "Large_Leduc_HDCFR_10"
    # name_str = "Large_Leduc_HDCFR_15"
    name_str = "Large_Leduc_HDCFR_20_2000"

    is_distributed = False
    if not is_distributed:
        N_LA_LEDUC_CFR = 1
    else:
        N_LA_LEDUC_CFR = 4

    ctrl = Driver(
        t_prof=TrainingProfile(
        checkpoint_freq=25,
        dim_c=3,
        name=name_str,
        path_data='./saved_data/' + name_str + '_' + t_str,

        n_learner_actor_workers=N_LA_LEDUC_CFR,
        nn_type="feedforward",

        periodic_restart=5,  # danger, time-consuming, very essential parameter

        sampler="learned_baseline",
        n_traversals_per_iter=2000, # *2

        # n_batches_per_iter_baseline=SDCFR_LEDUC_BASELINE_BATCHES,
        n_batches_per_iter_baseline=SDCFR_LEDUC_BASELINE_BATCHES,
        init_baseline_model="random",
        batch_size_baseline=int(512 / N_LA_LEDUC_CFR), # 512

        n_batches_adv_training=SDCFR_LEDUC_BATCHES,  # SDCFR_LEDUC_BATCHES
        mini_batch_size_adv=int(2048 / N_LA_LEDUC_CFR), # 2048
        max_buffer_size_adv=int(2e6 / N_LA_LEDUC_CFR),

        n_batches_avrg_training=4000,  # time-consuming， 4000
        mini_batch_size_avrg=2048, # 2048

        os_eps=0.5, # may be not useful for our sampling function design
        # game_cls=StandardLeduc,
        # game_cls=BigLeduc,
        # game_cls=LargeLeduc_10,
        # game_cls=LargeLeduc_15,
        game_cls=LargeLeduc_20,

        eval_modes_of_algo=(
            EvalAgent.EVAL_MODE_AVRG_NET,
        ),

        loss_baseline="mse",
        loss_avrg="weighted_mse",
        loss_adv="weighted_mse",

        DISTRIBUTED=is_distributed,

        device_inference="cpu",
        device_training="cpu",
        device_parameter_server="cpu",
        ),
        eval_methods={
            "br": 25, # 50
        },
        n_iterations=610
    )
    ctrl.run()

