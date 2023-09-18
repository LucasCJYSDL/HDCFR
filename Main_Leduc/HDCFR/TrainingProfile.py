import copy
import torch

from PokerRL.game import bet_sets
from PokerRL.game.wrappers import FlatLimitPokerEnvBuilder
from PokerRL.rl.base_cls.TrainingProfileBase import TrainingProfileBase

from HDCFR.workers.la.neural.HighMainPokerModule import HighMPMArgs
from HDCFR.workers.la.neural.LowMainPokerModule import LowMPMArgs
from HDCFR.workers.la.neural.BaselineMainPokerModule import BaselineMPMArgs

from HDCFR.workers.la.neural.HighAdvNet import HighAdvArgs
from HDCFR.workers.la.neural.LowAdvNet import LowAdvArgs
from HDCFR.workers.la.neural.HighAvrgNet import HighAvrgArgs
from HDCFR.workers.la.neural.LowAvrgNet import LowAvrgArgs
from HDCFR.workers.la.neural.BaselineNet import BaselineArgs

from HDCFR.workers.la.wrapper.AdvWrapper import HierAdvTrainingArgs
from HDCFR.workers.la.wrapper.AvrgWrapper import HierAvrgTrainingArgs
from HDCFR.workers.la.wrapper.BaselineWrapper import BaselineTrainingArgs

class TrainingProfile(TrainingProfileBase):

    def __init__(self,
                 # ------ General
                 name="",
                 log_verbose=False, # would show a lot of info if set as True, including loss
                 log_memory=False,
                 log_export_freq=1,
                 checkpoint_freq=99999999, # never save
                 eval_agent_export_freq=999999999, # never save
                 n_learner_actor_workers=8,
                 max_n_las_sync_simultaneously=10,
                 nn_type="feedforward",  # "recurrent" or "feedforward"

                 # ------ Computing
                 path_data=None,
                 local_crayon_server_docker_address="localhost",
                 device_inference="cpu", # for eval agent, and data generation
                 device_training="cpu", # for networks
                 device_parameter_server="cpu", # for the ps worker， it computes gradients so it's better on GPU
                 DISTRIBUTED=False,
                 CLUSTER=False,
                 DEBUGGING=False,

                 # ------ Env
                 game_cls=None,
                 n_seats=2,
                 agent_bet_set=bet_sets.B_2,
                 start_chips=None,
                 chip_randomness=(0, 0),
                 uniform_action_interpolation=False,
                 use_simplified_headsup_obs=True,

                 # ------ Evaluation
                 eval_modes_of_algo=None,
                 eval_stack_sizes=None,

                 # ------ General CFR params
                 n_traversals_per_iter=30000,
                 iter_weighting_exponent=1.0,
                 sampler=None,
                 os_eps=1,
                 periodic_restart=1,

                 # --- General Network Parameters
                 dim_h=64,
                 deep_net=True,
                 normalize_last_layer=True,

                 # --- MHA Parameters
                 dim_c=3, # important
                 dmodel=64, # important
                 mha_nhead=1, # important
                 mha_nlayers=1, # important
                 mha_nhid=64,
                 mha_dropout=0.2,

                 # --- Baseline Hyperparameters
                 max_buffer_size_baseline=2e5,
                 batch_size_baseline=512,
                 n_batches_per_iter_baseline=300,
                 loss_baseline="hdcfr_baseline_loss",
                 init_baseline_model="last", # TODO: change this to random, i.e., resetting for each iteration

                 # --- Adv Hyperparameters
                 n_batches_adv_training=5000,
                 init_adv_model="random",
                 mini_batch_size_adv=2048,
                 optimizer_adv="adam",
                 loss_adv="hdcfr_loss", # danger !!!!!!
                 lr_adv=0.001,
                 grad_norm_clipping_adv=1.0,
                 lr_patience_adv=999999999,
                 max_buffer_size_adv=2e6,

                 # for network reset
                 online=False,

                 # ------ SPECIFIC TO AVRG NET
                 n_batches_avrg_training=15000,
                 init_avrg_model="random",
                 mini_batch_size_avrg=2048,
                 loss_avrg="hdcfr_loss", # danger !!!!!!
                 optimizer_avrg="adam",
                 lr_avrg=0.001,
                 grad_norm_clipping_avrg=1.0,
                 lr_patience_avrg=999999999,
                 max_buffer_size_avrg=2e6, # very space-consuming !!!

                 # ------ SPECIFIC TO SINGLE
                 export_each_net=False,
                 eval_agent_max_strat_buf_size=None,

                 h2h_args = None,
                 rlbr_args=None
                 ):
        
        if nn_type == "feedforward":
            env_bldr_cls = FlatLimitPokerEnvBuilder

            mpm_args_adv_high = HighMPMArgs(dim_h, dim_c, dmodel, mha_nhead, mha_nlayers, mha_nhid, mha_dropout)
            mpm_args_adv_low = LowMPMArgs(deep=deep_net, dim=dim_h, normalize=normalize_last_layer, dim_c=dim_c)
            # mpm_args_adv_low = LowMPMArgs(deep=deep_net, dim=dim_h, normalize=normalize_last_layer, dmodel=dmodel)

            mpm_args_avrg_high = HighMPMArgs(dim_h, dim_c, dmodel, mha_nhead, mha_nlayers, mha_nhid, mha_dropout)
            mpm_args_avrg_low = LowMPMArgs(deep=deep_net, dim=dim_h, normalize=normalize_last_layer, dim_c=dim_c)
            # mpm_args_avrg_low = LowMPMArgs(deep=deep_net, dim=dim_h, normalize=normalize_last_layer, dmodel=dmodel)

            mpm_args_baseline = BaselineMPMArgs(deep=deep_net, dim=dim_h, normalize=normalize_last_layer, dim_c=dim_c)

        else:
            raise ValueError(nn_type)

        super().__init__(
            name=name,
            log_verbose=log_verbose,
            log_memory=log_memory,
            log_export_freq=log_export_freq,
            checkpoint_freq=checkpoint_freq,
            eval_agent_export_freq=eval_agent_export_freq,

            game_cls=game_cls,
            env_bldr_cls=env_bldr_cls,
            start_chips=start_chips,

            eval_modes_of_algo=eval_modes_of_algo,
            eval_stack_sizes=eval_stack_sizes,

            path_data=path_data,
            DEBUGGING=DEBUGGING,
            DISTRIBUTED=DISTRIBUTED,
            CLUSTER=CLUSTER,
            device_inference=device_inference, # used for EvalAgent
            local_crayon_server_docker_address=local_crayon_server_docker_address,

            module_args={
                "env": game_cls.ARGS_CLS(
                    n_seats=n_seats,
                    starting_stack_sizes_list=[start_chips for _ in range(n_seats)], # [None, None]
                    bet_sizes_list_as_frac_of_pot=copy.deepcopy(agent_bet_set),
                    stack_randomization_range=chip_randomness,
                    use_simplified_headsup_obs=use_simplified_headsup_obs,
                    uniform_action_interpolation=uniform_action_interpolation
                ),
                "adv_training": HierAdvTrainingArgs(
                    high_adv_net_args=HighAdvArgs(
                        mpm_args=mpm_args_adv_high,
                        n_units_final=dim_h
                    ),
                    low_adv_net_args=LowAdvArgs(
                        mpm_args=mpm_args_adv_low,
                        n_units_final=dim_h
                    ),
                    n_batches_adv_training=n_batches_adv_training, # time
                    init_adv_model=init_adv_model, # time
                    batch_size=mini_batch_size_adv,
                    optim_str=optimizer_adv,
                    loss_str=loss_adv,
                    lr=lr_adv,
                    grad_norm_clipping=grad_norm_clipping_adv,
                    device_training=device_training, # time
                    max_buffer_size=max_buffer_size_adv,
                    lr_patience=lr_patience_adv,
                ),
                "avrg_training": HierAvrgTrainingArgs(
                    high_avrg_net_args=HighAvrgArgs(
                        mpm_args=mpm_args_avrg_high,
                        n_units_final=dim_h
                    ),
                    low_avrg_net_args=LowAvrgArgs(
                        mpm_args=mpm_args_avrg_low,
                        n_units_final=dim_h
                    ),
                    n_batches_avrg_training=n_batches_avrg_training, # time
                    init_avrg_model=init_avrg_model, # time
                    batch_size=mini_batch_size_avrg,
                    loss_str=loss_avrg,
                    optim_str=optimizer_avrg,
                    lr=lr_avrg,
                    grad_norm_clipping=grad_norm_clipping_avrg,
                    device_training=device_training, # time
                    max_buffer_size=max_buffer_size_avrg,
                    lr_patience=lr_patience_avrg,
                ),
                "mccfr_baseline": BaselineTrainingArgs(  # trainer
                    net_args=BaselineArgs(  # network
                        mpm_args=mpm_args_baseline,
                        n_units_final=dim_h
                    ),
                    max_buffer_size=max_buffer_size_baseline,
                    batch_size=batch_size_baseline,
                    n_batches_per_iter_baseline=n_batches_per_iter_baseline, # time
                    loss_str=loss_baseline,
                    device_training=device_training,  # time
                    init_model=init_baseline_model, # whether to reset the baseline for each iteration
                    dim_c=dim_c,
                ),
                "h2h": h2h_args,
                "rlbr": rlbr_args,
            },
        )

        self.dim_c = dim_c
        self.online = online
        self.nn_type = nn_type
        self.n_traversals_per_iter = int(n_traversals_per_iter) # time
        self.sampler = sampler
        self.os_eps = os_eps
        self.periodic_restart = periodic_restart # time

        self.iter_weighting_exponent = iter_weighting_exponent

        # SINGLE
        self.export_each_net = export_each_net
        self.eval_agent_max_strat_buf_size = eval_agent_max_strat_buf_size # how many adv nets to store

        # Different for dist and local
        if DISTRIBUTED or CLUSTER:
            print("Running with ", n_learner_actor_workers, "LearnerActor Workers.")
            self.n_learner_actors = n_learner_actor_workers
        else:
            self.n_learner_actors = 1
        self.max_n_las_sync_simultaneously = max_n_las_sync_simultaneously

        assert isinstance(device_parameter_server, str), "Please pass a string (either 'cpu' or 'cuda')!"
        self.device_parameter_server = torch.device(device_parameter_server)
