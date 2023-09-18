import numpy as np
import datetime

from HYPERS import *
from PokerRL import Flop3Holdem, LargeFlop3Holdem

from HDCFR.TrainingProfile import TrainingProfile
from HDCFR.workers.driver.Driver import Driver
from HDCFR.EvalAgent import EvalAgent

if __name__ == '__main__':
    t = datetime.datetime.now()
    t_str = t.strftime('%Y-%m-%d-%H-%M-%S')
    name_str = "FHP_HDCFR"
    # name_str = "LargeFHP_HDCFR"

    DIM_C = 3

    ctrl = Driver(t_prof=TrainingProfile(
        checkpoint_freq=50,
        name=name_str,
        path_data='./saved_data/' + name_str + '_' + t_str,
        dim_c=DIM_C,

        n_traversals_per_iter=(SDCFR_FHP_TRAVERSALS_OS / N_LA_FHP_CFR) * min(DIM_C, 2), # *2
        sampler="learned_baseline",
        os_eps=OS_EPS,

        n_batches_per_iter_baseline=2000,
        init_baseline_model="random",
        batch_size_baseline=int(2048 / N_LA_FHP_CFR),

        game_cls=Flop3Holdem,
        # game_cls=LargeFlop3Holdem,
        n_batches_adv_training=SDCFR_FHP_BATCHES,
        n_learner_actor_workers=N_LA_FHP_CFR,
        mini_batch_size_adv=int(SDCFR_FHP_BATCH_SIZE / N_LA_FHP_CFR),
        max_buffer_size_adv=int(4e7 / N_LA_FHP_CFR),
        
        DISTRIBUTED=True,
        rlbr_args=DIST_RLBR_ARGS_games,
        nn_type="feedforward",
        eval_modes_of_algo=(
            EvalAgent.EVAL_MODE_AVRG_NET,
        ),

        loss_baseline="mse",
        loss_avrg="weighted_mse",
        loss_adv="weighted_mse",

        device_inference="cpu",
        device_training="cpu",
        device_parameter_server="cpu",
        periodic_restart=5
        ),
        eval_methods={'h2h_eval': -1}, # {'h2h_eval': -1},
        n_iterations=1000
    )
    ctrl.run()