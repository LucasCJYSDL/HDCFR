import numpy as np
import datetime

from NFSP.TrainingProfile import TrainingProfile
from NFSP.workers.driver.Driver import Driver
from PokerRL import StandardLeduc, BigLeduc, LargeLeduc

if __name__ == '__main__':

    t = datetime.datetime.now()
    t_str = t.strftime('%Y-%m-%d-%H-%M-%S')
    # name_str = "Leduc_NFSP"
    # name_str = "Big_Leduc_NFSP"
    name_str = "Large_Leduc_NFSP"
    ctrl = Driver(t_prof=TrainingProfile(
        checkpoint_freq=10000,
        name=name_str,
        path_data='./saved_data/'+name_str+'_'+t_str,
        # game_cls=StandardLeduc,
        # game_cls=BigLeduc,
        game_cls=LargeLeduc,
        n_steps_per_iter_per_la=128,
        target_net_update_freq=300,
        min_prob_add_res_buf=0,
        lr_avg=0.01,
        lr_br=0.1,
        DISTRIBUTED=False,
        # device_inference="cuda",
        # device_parameter_server="cuda",
        device_inference="cpu",
        device_training="cuda:1",
        device_parameter_server="cpu",
    ),
        eval_methods={"br": 10000},
        n_iterations=10000000)
    ctrl.run()
