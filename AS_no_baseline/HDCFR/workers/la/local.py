import os
import psutil

from PokerRL.rl import rl_util
from PokerRL.rl.base_cls.workers.WorkerBase import WorkerBase

from HDCFR.workers.la.wrapper.AdvWrapper import HierAdvWrapper
from HDCFR.workers.la.buffer.AdvReservoirBuffer import AdvReservoirBuffer
from HDCFR.workers.la.wrapper.AvrgWrapper import HierAvrgWrapper
from HDCFR.workers.la.buffer.AvrgReservoirBuffer import AvrgReservoirBuffer
from HDCFR.workers.la.wrapper.BaselineWrapper import BaselineWrapper
from HDCFR.workers.la.buffer.BaselineBuffer import BaselineBuffer
from HDCFR.workers.la.sampler.OutcomeSampler import OutcomeSampler
from HDCFR.IterationStrategy import IterationStrategy


class LearnerActor(WorkerBase):

    def __init__(self, t_prof, worker_id, chief_handle):
        super().__init__(t_prof=t_prof)

        # set up the regret network
        self._adv_args = t_prof.module_args["adv_training"]

        self._env_bldr = rl_util.get_env_builder(t_prof=t_prof)
        self._id = worker_id
        self._chief_handle = chief_handle

        self._adv_buffers = [
            AdvReservoirBuffer(owner=p, env_bldr=self._env_bldr, max_size=self._adv_args.max_buffer_size,
                               nn_type=t_prof.nn_type, iter_weighting_exponent=self._t_prof.iter_weighting_exponent,
                               t_prof=t_prof)
            for p in range(self._t_prof.n_seats)
        ]

        self._adv_wrappers = [
            HierAdvWrapper(owner=p,
                           env_bldr=self._env_bldr,
                           adv_training_args=self._adv_args,
                           device=self._adv_args.device_training)
            for p in range(self._t_prof.n_seats)
        ]

        # set up the average network

        self._avrg_args = t_prof.module_args["avrg_training"]

        self._avrg_buffers = [
            AvrgReservoirBuffer(owner=p, env_bldr=self._env_bldr, max_size=self._avrg_args.max_buffer_size,
                                nn_type=t_prof.nn_type, iter_weighting_exponent=self._t_prof.iter_weighting_exponent,
                                t_prof=t_prof)
            for p in range(self._t_prof.n_seats)
        ]

        self._avrg_wrappers = [
            HierAvrgWrapper(owner=p,
                            env_bldr=self._env_bldr,
                            avrg_training_args=self._avrg_args,
                            device=self._avrg_args.device_training)
            for p in range(self._t_prof.n_seats)
        ]

        # set up the baseline network
        if self._t_prof.sampler.lower() == "learned_baseline":
            self._baseline_args = t_prof.module_args["mccfr_baseline"]
            self._baseline_wrapper = BaselineWrapper(env_bldr=self._env_bldr,
                                                     baseline_args=self._baseline_args,
                                                     device=self._baseline_args.device_training)

            self._baseline_buf = BaselineBuffer(owner=None, env_bldr=self._env_bldr,
                                                max_size=self._baseline_args.max_buffer_size, nn_type=t_prof.nn_type, dim_c=t_prof.dim_c)

            self._data_sampler = OutcomeSampler(
                env_bldr=self._env_bldr, adv_buffers=self._adv_buffers, eps=self._t_prof.os_eps,
                baseline_net=self._baseline_wrapper, baseline_buf=self._baseline_buf,
                avrg_buffers=self._avrg_buffers, t_prof=t_prof)
        else:
            raise ValueError("Currently we don't support", self._t_prof.sampler.lower(), "sampling.")

        # others
        if self._t_prof.log_memory:
            self._exp_mem_usage = self._ray.get(self._ray.remote(self._chief_handle.create_experiment,
                                                self._t_prof.name + "_LA" + str(worker_id) + "_Memory_Usage"))
            self._exps_adv_buffer_size = self._ray.get(
                [
                    self._ray.remote(self._chief_handle.create_experiment,
                                     self._t_prof.name + "_LA" + str(worker_id) + "_P" + str(p) + "_ADV_BufSize")
                    for p in range(self._t_prof.n_seats)
                ]
            )

            self._exps_avrg_buffer_size = self._ray.get(
                [
                    self._ray.remote(self._chief_handle.create_experiment,
                                     self._t_prof.name + "_LA" + str(worker_id) + "_P" + str(p) + "_AVRG_BufSize")
                    for p in range(self._t_prof.n_seats)
                ]
            )


    def get_loss_last_batch_adv(self, p_id):
        return self._adv_wrappers[p_id].get_loss_last_batch() # (high_loss_batch, low_loss_batch)

    def get_loss_last_batch_avrg(self, p_id):
        return self._avrg_wrappers[p_id].get_loss_last_batch()

    def get_loss_last_batch_baseline(self):
        return self._baseline_wrapper.loss_last_batch

    def get_high_adv_grads(self, p_id):
        return self._ray.grads_to_numpy(self._adv_wrappers[p_id].get_high_grads(buffer=self._adv_buffers[p_id]))

    def get_low_adv_grads(self, p_id):
        return self._ray.grads_to_numpy(self._adv_wrappers[p_id].get_low_grads(buffer=self._adv_buffers[p_id]))

    def get_high_avrg_grads(self, p_id):
        return self._ray.grads_to_numpy(self._avrg_wrappers[p_id].get_high_grads(buffer=self._avrg_buffers[p_id]))

    def get_low_avrg_grads(self, p_id):
        return self._ray.grads_to_numpy(self._avrg_wrappers[p_id].get_low_grads(buffer=self._avrg_buffers[p_id]))

    def get_baseline_grads(self):
        return self._ray.grads_to_numpy(self._baseline_wrapper.get_grads_one_batch_from_buffer(buffer=self._baseline_buf))

    def update(self, adv_state_dicts=None, avrg_state_dicts=None, baseline_state_dict=None):

        baseline_state_dict = baseline_state_dict[0]  # wrapped bc of object id stuff
        if baseline_state_dict is not None:
            self._baseline_wrapper.load_net_state_dict(state_dict=self._ray.state_dict_to_torch(self._ray.get(baseline_state_dict),
                                                       device=self._baseline_wrapper.device))

        for p_id in range(self._t_prof.n_seats):
            if adv_state_dicts[p_id] is not None:
                self._adv_wrappers[p_id].load_net_state_dict(
                    state_dict=(self._ray.state_dict_to_torch(self._ray.get(adv_state_dicts[p_id][0]), device=self._adv_wrappers[p_id].device),\
                                self._ray.state_dict_to_torch(self._ray.get(adv_state_dicts[p_id][1]), device=self._adv_wrappers[p_id].device))
                )

            if avrg_state_dicts[p_id] is not None:
                self._avrg_wrappers[p_id].load_net_state_dict(
                    state_dict=(self._ray.state_dict_to_torch(self._ray.get(avrg_state_dicts[p_id][0]), device=self._avrg_wrappers[p_id].device), \
                                self._ray.state_dict_to_torch(self._ray.get(avrg_state_dicts[p_id][1]), device=self._avrg_wrappers[p_id].device))
                )


    def generate_data(self, traverser, cfr_iter):
        iteration_strats = [
            IterationStrategy(t_prof=self._t_prof, env_bldr=self._env_bldr, owner=p,
                              device=self._t_prof.device_inference, cfr_iter=cfr_iter)
            for p in range(self._t_prof.n_seats)
        ]

        for s in iteration_strats: # the strategy at the current iteration can be derived from the advantage function
            s.load_net_state_dict(state_dict=self._adv_wrappers[s.owner].net_state_dict()) # their devices are different

        self._data_sampler.generate(n_traversals=self._t_prof.n_traversals_per_iter,
                                    traverser=traverser,
                                    iteration_strats=iteration_strats,
                                    cfr_iter=cfr_iter,
                                   )

        # Log after both players generated data
        if self._t_prof.log_memory and traverser == 1 and (cfr_iter % 3 == 0):
            for p in range(self._t_prof.n_seats):
                self._ray.remote(self._chief_handle.add_scalar,
                                 self._exps_adv_buffer_size[p], "Debug/BufferSize", cfr_iter,
                                 self._adv_buffers[p].size)

                self._ray.remote(self._chief_handle.add_scalar,
                                 self._exps_avrg_buffer_size[p], "Debug/BufferSize", cfr_iter,
                                 self._avrg_buffers[p].size)

            process = psutil.Process(os.getpid())
            self._ray.remote(self._chief_handle.add_scalar,
                             self._exp_mem_usage, "Debug/MemoryUsage/LA", cfr_iter,
                             process.memory_info().rss)

        return self._data_sampler.total_node_count_traversed

    def reset_baseline_buffer(self):
        self._baseline_buf.reset()

    def baseline_buffer_to_list(self):
        self._baseline_buf.to_list()

    def get_target_b(self, cfr_iter):
        iteration_strats = [
            IterationStrategy(t_prof=self._t_prof, env_bldr=self._env_bldr, owner=p,
                              device=self._t_prof.device_inference, cfr_iter=cfr_iter)
            for p in range(self._t_prof.n_seats)
        ]

        for s in iteration_strats:  # the strategy at the current iteration can be derived from the advantage function
            s.load_net_state_dict(
                state_dict=self._adv_wrappers[s.owner].net_state_dict())  # their devices are different

        self._data_sampler.get_target_b(iter_starts_tp1=iteration_strats)