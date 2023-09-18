import numpy as np
import datetime

from HYPERS import *
from NFSP.TrainingProfile import TrainingProfile
from NFSP.workers.driver.Driver import Driver
from PokerRL import Flop3Holdem, LargeFlop3Holdem

if __name__ == '__main__':
    t = datetime.datetime.now()
    t_str = t.strftime('%Y-%m-%d-%H-%M-%S')
    name_str = "FHP_NFSP"
    # name_str = "LargeFHP_NFSP"

    N_LA_FHP_NFSP = 1

    ctrl = Driver(t_prof=TrainingProfile(
        checkpoint_freq=50000,
        name=name_str,
        path_data='./saved_data/' + name_str + '_' + t_str,
        game_cls=Flop3Holdem,
        # game_cls=LargeFlop3Holdem,
        eps_const=0.005,
        eps_start=0.08,
        target_net_update_freq=1000,
        min_prob_add_res_buf=0.25,
        lr_avg=0.01,
        lr_br=0.1,

        n_learner_actor_workers=N_LA_FHP_NFSP,
        res_buf_size_each_la=int(2e7 / N_LA_FHP_NFSP),
        cir_buf_size_each_la=int(6e5 / N_LA_FHP_NFSP),
        n_steps_per_iter_per_la=int(256 / N_LA_FHP_NFSP),
        mini_batch_size_br_per_la=int(256 / N_LA_FHP_NFSP),
        mini_batch_size_avg_per_la=int(256 / N_LA_FHP_NFSP),

        DISTRIBUTED=False,
        rlbr_args=DIST_RLBR_ARGS_games,
        device_inference="cpu",
        device_training="cpu",
        device_parameter_server="cpu",

    ),
        eval_methods={'h2h_eval': -1},
        n_iterations=1000000)
    ctrl.run()
