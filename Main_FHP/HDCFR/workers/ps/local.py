import os
import pickle

import psutil
from torch.optim import lr_scheduler

from PokerRL.rl import rl_util
from PokerRL.rl.base_cls.workers.ParameterServerBase import ParameterServerBase

from HDCFR.EvalAgent import EvalAgent
from HDCFR.workers.la.neural.HighAdvNet import HighAdvVet
from HDCFR.workers.la.neural.LowAdvNet import LowAdvNet
from HDCFR.workers.la.neural.HighAvrgNet import HighAvrgNet
from HDCFR.workers.la.neural.LowAvrgNet import LowAvrgNet
from HDCFR.workers.la.neural.BaselineNet import BaselineNet

class ParameterServer(ParameterServerBase):

    def __init__(self, t_prof, owner, chief_handle):
        super().__init__(t_prof=t_prof, chief_handle=chief_handle)

        self.owner = owner

        # regret net
        self._adv_args = t_prof.module_args["adv_training"]
        self._high_adv_net, self._low_adv_net = self._get_new_adv_net()
        self._high_adv_optim, self._high_adv_lr_scheduler,\
        self._low_adv_optim, self._low_adv_lr_scheduler = self._get_new_adv_optim()

        if self._t_prof.log_memory:
            self._exp_mem_usage = self._ray.get(
                self._ray.remote(self._chief_handle.create_experiment,
                                 self._t_prof.name + "_PS" + str(owner) + "_Memory_Usage"))

        # average net
        self._AVRG = EvalAgent.EVAL_MODE_AVRG_NET in self._t_prof.eval_modes_of_algo
        self._SINGLE = EvalAgent.EVAL_MODE_SINGLE in self._t_prof.eval_modes_of_algo

        assert self._AVRG and not self._SINGLE, "The single mode is not part of our algorthm design."

        self._avrg_args = t_prof.module_args["avrg_training"]
        self._high_avrg_net, self._low_avrg_net = self._get_new_avrg_net()
        self._high_avrg_optim, self._high_avrg_lr_scheduler,\
        self._low_avrg_optim, self._low_avrg_lr_scheduler = self._get_new_avrg_optim()

        # baseline
        assert self._t_prof.sampler == "learned_baseline"
        if owner == 0: # only maintain baseline for player 0
            self._baseline_args = t_prof.module_args["mccfr_baseline"]
            self._baseline_net = self._get_new_baseline_net()
            self._baseline_optim = self._get_new_baseline_optim()


    def _get_new_adv_net(self):

        high_net =  HighAdvVet(env_bldr=self._env_bldr, args=self._adv_args.high_adv_net_args, device=self._device)
        low_net = LowAdvNet(env_bldr=self._env_bldr, args=self._adv_args.low_adv_net_args, device=self._device)
        low_net.set_option_emb(high_net.get_option_emb())

        return high_net, low_net

    def _get_new_adv_optim(self):
        high_opt = rl_util.str_to_optim_cls(self._adv_args.optim_str)(self._high_adv_net.parameters(), lr=self._adv_args.lr)
        high_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=high_opt,
                                                        threshold=0.001,
                                                        factor=0.5,
                                                        patience=self._adv_args.lr_patience,
                                                        min_lr=0.00002)

        low_opt = rl_util.str_to_optim_cls(self._adv_args.optim_str)(self._low_adv_net.parameters(), lr=self._adv_args.lr)
        low_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=low_opt,
                                                       threshold=0.001,
                                                       factor=0.5,
                                                       patience=self._adv_args.lr_patience,
                                                       min_lr=0.00002)

        return high_opt, high_scheduler, low_opt, low_scheduler

    def _get_new_avrg_net(self):
        high_net = HighAvrgNet(avrg_net_args=self._avrg_args.high_avrg_net_args, env_bldr=self._env_bldr, device=self._device)
        low_net = LowAvrgNet(avrg_net_args=self._avrg_args.low_avrg_net_args, env_bldr=self._env_bldr, device=self._device)
        low_net.set_option_emb(high_net.get_option_emb())

        return high_net, low_net

    def _get_new_avrg_optim(self):
        high_opt = rl_util.str_to_optim_cls(self._avrg_args.optim_str)(self._high_avrg_net.parameters(), lr=self._avrg_args.lr)
        high_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=high_opt,
                                                        threshold=0.0001,
                                                        factor=0.5,
                                                        patience=self._avrg_args.lr_patience,
                                                        min_lr=0.00002)

        low_opt = rl_util.str_to_optim_cls(self._avrg_args.optim_str)(self._low_avrg_net.parameters(), lr=self._avrg_args.lr)
        low_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=low_opt,
                                                       threshold=0.0001,
                                                       factor=0.5,
                                                       patience=self._avrg_args.lr_patience,
                                                       min_lr=0.00002)

        return high_opt, high_scheduler, low_opt, low_scheduler

    def _get_new_baseline_net(self):
        return BaselineNet(env_bldr=self._env_bldr, args=self._baseline_args.net_args, device=self._device)

    def _get_new_baseline_optim(self):
        opt = rl_util.str_to_optim_cls(self._baseline_args.optim_str)(self._baseline_net.parameters(),
                                                                      lr=self._baseline_args.lr)
        return opt

    def get_adv_weights(self):
        self._high_adv_net.zero_grad()
        self._low_adv_net.zero_grad()
        return (self._ray.state_dict_to_numpy(self._high_adv_net.state_dict()), \
                self._ray.state_dict_to_numpy(self._low_adv_net.state_dict()))

    def get_avrg_weights(self):
        self._high_avrg_net.zero_grad()
        self._low_avrg_net.zero_grad()
        return (self._ray.state_dict_to_numpy(self._high_avrg_net.state_dict()),
                self._ray.state_dict_to_numpy(self._low_avrg_net.state_dict()))

    def get_baseline_weights(self):
        self._baseline_net.zero_grad()
        return self._ray.state_dict_to_numpy(self._baseline_net.state_dict())

    def step_scheduler_adv(self, high_loss=None, low_loss=None):
        if high_loss:
            self._high_adv_lr_scheduler.step(high_loss)
        if low_loss:
            self._low_adv_lr_scheduler.step(low_loss)

    def step_scheduler_avrg(self, high_loss=None, low_loss=None):
        if high_loss:
            self._high_avrg_lr_scheduler.step(high_loss)
        if low_loss:
            self._low_avrg_lr_scheduler.step(low_loss)

    def checkpoint(self, curr_step):
        state = {
            "adv_net": (self._high_adv_net.state_dict(), self._low_adv_net.state_dict()),
            "adv_optim": (self._high_adv_optim.state_dict(), self._low_adv_optim.state_dict()),
            "adv_lr_sched": (self._high_adv_lr_scheduler.state_dict(), self._low_adv_lr_scheduler.state_dict()),
            "seat_id": self.owner,
        }

        state["avrg_net"] = (self._high_avrg_net.state_dict(), self._low_avrg_net.state_dict())
        state["avrg_optim"] = (self._high_avrg_optim.state_dict(), self._low_avrg_optim.state_dict())
        state["avrg_lr_sched"] = (self._high_avrg_lr_scheduler.state_dict(), self._low_avrg_lr_scheduler.state_dict())

        if self.owner == 0:
            state["baseline_net"] = self._baseline_net.state_dict()
            state["baseline_optim"] = self._baseline_optim.state_dict()

        with open(self._get_checkpoint_file_path(name=self._t_prof.name, step=curr_step,
                                                 cls=self.__class__, worker_id="P" + str(self.owner)), "wb") as pkl_file:
            pickle.dump(obj=state, file=pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, name_to_load, step):
        with open(self._get_checkpoint_file_path(name=name_to_load, step=step,
                                                 cls=self.__class__, worker_id="P" + str(self.owner)), "rb") as pkl_file:
            state = pickle.load(pkl_file)

            assert self.owner == state["seat_id"]

        self._high_adv_net.load_state_dict(state["adv_net"][0])
        self._low_adv_net.load_state_dict(state["adv_net"][1])
        self._high_adv_optim.load_state_dict(state["adv_optim"][0])
        self._low_adv_optim.load_state_dict(state["adv_optim"][1])
        self._high_adv_lr_scheduler.load_state_dict(state["adv_lr_sched"][0])
        self._low_adv_lr_scheduler.load_state_dict(state["adv_lr_sched"][1])

        self._high_avrg_net.load_state_dict(state["avrg_net"][0])
        self._low_avrg_net.load_state_dict(state["avrg_net"][1])
        self._high_avrg_optim.load_state_dict(state["avrg_optim"][0])
        self._low_avrg_optim.load_state_dict(state["avrg_optim"][1])
        self._high_avrg_lr_scheduler.load_state_dict(state["avrg_lr_sched"][0])
        self._low_avrg_lr_scheduler.load_state_dict(state["avrg_lr_sched"][1])

        if self.owner == 0:
            self._baseline_net.load_state_dict(state["baseline_net"])
            self._baseline_optim.load_state_dict(state["baseline_optim"])


    def reset_adv_net(self, cfr_iter):
        if self._adv_args.init_adv_model == "last":
            self._high_adv_net.zero_grad()
            self._low_adv_net.zero_grad()
            if not self._t_prof.online:
                self._high_adv_optim, self._high_adv_lr_scheduler,\
                self._low_adv_optim, self._low_adv_lr_scheduler = self._get_new_adv_optim()

        elif self._adv_args.init_adv_model == "random":
            self._high_adv_net, self._low_adv_net = self._get_new_adv_net()
            self._high_adv_optim, self._high_adv_lr_scheduler, \
                self._low_adv_optim, self._low_adv_lr_scheduler = self._get_new_adv_optim()

        else:
            raise ValueError(self._adv_args.init_adv_model)

        if self._t_prof.log_memory and (cfr_iter % 3 == 0):
            # Logs
            process = psutil.Process(os.getpid())
            self._ray.remote(self._chief_handle.add_scalar,
                             self._exp_mem_usage, "Debug/MemoryUsage/PS", cfr_iter, process.memory_info().rss)

    def reset_baseline_net(self):
        if self._baseline_args.init_model == "last":
            self._baseline_net.zero_grad()
            if not self._t_prof.online:
                self._baseline_optim = self._get_new_baseline_optim()
        else:
            self._baseline_net = self._get_new_baseline_net()
            self._baseline_optim = self._get_new_baseline_optim()


    def reset_avrg_net(self):
        if self._avrg_args.init_avrg_model == "last":
            self._high_avrg_net.zero_grad()
            self._low_avrg_net.zero_grad()
            if not self._t_prof.online:
                self._high_avrg_optim, self._high_avrg_lr_scheduler, \
                    self._low_avrg_optim, self._low_avrg_lr_scheduler = self._get_new_avrg_optim()

        elif self._avrg_args.init_avrg_model == "random":
            self._high_avrg_net, self._low_avrg_net = self._get_new_avrg_net()
            self._high_avrg_optim, self._high_avrg_lr_scheduler, \
                self._low_avrg_optim, self._low_avrg_lr_scheduler = self._get_new_avrg_optim()

        else:
            raise ValueError(self._avrg_args.init_avrg_model)

    def apply_grads_high_adv(self, list_of_grads):
        self._apply_grads(list_of_grads=list_of_grads, optimizer=self._high_adv_optim, net=self._high_adv_net,
                          grad_norm_clip=self._adv_args.grad_norm_clipping)

    def apply_grads_low_adv(self, list_of_grads):
        self._apply_grads(list_of_grads=list_of_grads, optimizer=self._low_adv_optim, net=self._low_adv_net,
                          grad_norm_clip=self._adv_args.grad_norm_clipping)

    def apply_grads_high_avrg(self, list_of_grads):
        self._apply_grads(list_of_grads=list_of_grads, optimizer=self._high_avrg_optim, net=self._high_avrg_net,
                          grad_norm_clip=self._avrg_args.grad_norm_clipping)

    def apply_grads_low_avrg(self, list_of_grads):
        self._apply_grads(list_of_grads=list_of_grads, optimizer=self._low_avrg_optim, net=self._low_avrg_net,
                          grad_norm_clip=self._avrg_args.grad_norm_clipping)

    def apply_grads_baseline(self, list_of_grads):
        self._apply_grads(list_of_grads=list_of_grads, optimizer=self._baseline_optim, net=self._baseline_net,
                          grad_norm_clip=self._baseline_args.grad_norm_clipping)