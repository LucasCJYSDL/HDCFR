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

    is_distributed = False
    if not is_distributed:
        N_LA_LEDUC_CFR = 1
    else:
        N_LA_LEDUC_CFR = 4

    iter_import = 100
    name_str_import = "Leduc_HDCFR"
    t_str_import = "2023-09-08-10-56-48"

    # iter_import = 100
    # name_str_import = "Large_Leduc_HDCFR_10"
    # t_str_import = "2023-09-05-11-29-42"

    # iter_import = 375
    # name_str_import = "Large_Leduc_HDCFR_15"
    # t_str_import = "2023-09-05-15-37-35"

    # iter_import = 600
    # name_str_import = "Large_Leduc_HDCFR_20"
    # t_str_import = "2023-08-27-16-23-45"

    #  is_fixed = True
    # is_fixed_baseline = True
    # init_baseline = True

    is_fixed = False
    is_fixed_baseline = False
    init_baseline = True


    ctrl = Driver(
        t_prof=TrainingProfile(
        checkpoint_freq=25,
        dim_c=3,
        name=name_str_import,
        path_data='./saved_data/' + name_str_import + '_' + t_str_import,
        # name=name_str,
        # path_data='./saved_data/' + name_str + '_' + t_str,

        n_learner_actor_workers=N_LA_LEDUC_CFR,
        nn_type="feedforward",

        periodic_restart=5,  # danger, time-consuming, very essential parameter

        sampler="learned_baseline",
        n_traversals_per_iter=SDCFR_LEDUC_TRAVERSALS_OS / N_LA_LEDUC_CFR * 2, # *2

        n_batches_per_iter_baseline=SDCFR_LEDUC_BASELINE_BATCHES,
        init_baseline_model="random",
        batch_size_baseline=int(512 / N_LA_LEDUC_CFR), # 512

        n_batches_adv_training=SDCFR_LEDUC_BATCHES,  # SDCFR_LEDUC_BATCHES
        mini_batch_size_adv=int(2048 / N_LA_LEDUC_CFR), # 2048
        max_buffer_size_adv=int(2e6 / N_LA_LEDUC_CFR),

        n_batches_avrg_training=4000,  # time-consuming， 4000
        mini_batch_size_avrg=2048, # 2048

        os_eps=OS_EPS, # may be not useful for our sampling function design
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

        is_fixed=is_fixed,
        init_baseline=init_baseline,
        is_fixed_baseline=is_fixed_baseline
        ),
        eval_methods={
            "br": 25, # 50
        },
        n_iterations=610,
        iteration_to_import=iter_import,
        name_to_import=name_str_import
    )
    ctrl.run()

