import time
from tqdm import tqdm
from PokerRL.rl.base_cls.HighLevelAlgoBase import HighLevelAlgoBase as _HighLevelAlgoBase


class HighLevelAlgo(_HighLevelAlgoBase):

    def __init__(self, t_prof, la_handles, ps_handles, chief_handle):
        super().__init__(t_prof=t_prof, chief_handle=chief_handle, la_handles=la_handles)
        self._ps_handles = ps_handles
        self._all_p_aranged = list(range(self._t_prof.n_seats))

        self._baseline_args = t_prof.module_args["mccfr_baseline"]
        self._adv_args = t_prof.module_args["adv_training"]
        self._avrg_args = t_prof.module_args["avrg_training"]

        self._exp_states_traversed = self._ray.get(self._ray.remote(self._chief_handle.create_experiment,
                                                                    self._t_prof.name + "_States_traversed"))

    def init(self):

        self._update_leaner_actors(update_adv_for_plyrs=self._all_p_aranged,
                                   update_avrg_for_plyrs=self._all_p_aranged)

    def _update_leaner_actors(self, update_adv_for_plyrs=None, update_avrg_for_plyrs=None, update_baseline=None):
        """

        Args:
            update_adv_for_plyrs (list):         list of player_ids to update adv for
            update_avrg_for_plyrs (list):        list of player_ids to update avrg for
        """

        assert isinstance(update_adv_for_plyrs, list) or update_adv_for_plyrs is None
        assert isinstance(update_avrg_for_plyrs, list) or update_avrg_for_plyrs is None
        assert isinstance(update_baseline, bool) or update_baseline is None

        _update_adv_per_p = [
            True if (update_adv_for_plyrs is not None) and (p in update_adv_for_plyrs) else False
            for p in range(self._t_prof.n_seats)
        ]
        _update_avrg_per_p = [
            True if (update_avrg_for_plyrs is not None) and (p in update_avrg_for_plyrs) else False
            for p in range(self._t_prof.n_seats)
        ]

        la_batches = []
        n = len(self._la_handles)
        c = 0
        while n > c:
            s = min(n, c + self._t_prof.max_n_las_sync_simultaneously)
            la_batches.append(self._la_handles[c:s])
            if type(la_batches[-1]) is not list:
                la_batches[-1] = [la_batches[-1]]
            c = s

        w_adv = [None for _ in range(self._t_prof.n_seats)]
        w_avrg = [None for _ in range(self._t_prof.n_seats)]

        # only one baseline is maintained, and it is assigned to player 0 by default
        w_baseline = [None if not update_baseline else self._ray.remote(self._ps_handles[0].get_baseline_weights)]

        for p_id in range(self._t_prof.n_seats):
            w_adv[p_id] = None if not _update_adv_per_p[p_id] else self._ray.remote(
                self._ps_handles[p_id].get_adv_weights)

            w_avrg[p_id] = None if not _update_avrg_per_p[p_id] else self._ray.remote(
                self._ps_handles[p_id].get_avrg_weights)

        for batch in la_batches:
            self._ray.wait([
                self._ray.remote(la.update,
                                 w_adv,
                                 w_avrg,
                                 w_baseline)
                for la in batch
            ])

    # train average net

    def train_average_nets(self, cfr_iter):
        print("Training Average Nets...")
        t_computation_avrg = 0.0
        t_syncing_avrg = 0.0
        for p in range(self._t_prof.n_seats):
            _c, _s = self._train_avrg(p_id=p, cfr_iter=cfr_iter)
            t_computation_avrg += _c
            t_syncing_avrg += _s

        return {
            "t_computation_avrg": t_computation_avrg,
            "t_syncing_avrg": t_syncing_avrg,
        }

    def _train_avrg(self, p_id, cfr_iter):
        t_computation = 0.0
        t_syncing = 0.0

        # For logging the loss to see convergence in Tensorboard
        if self._t_prof.log_verbose:
            exp_high_loss_each_p = [
                self._ray.remote(self._chief_handle.create_experiment,
                                 self._t_prof.name + "_AverageNet_High_Loss_P" + str(p) + "_I" + str(cfr_iter)
                )
                for p in range(self._t_prof.n_seats)
            ]
            exp_low_loss_each_p = [
                self._ray.remote(
                    self._chief_handle.create_experiment,
                    self._t_prof.name + "_AverageNet_Low_Loss_P" + str(p) + "_I" + str(cfr_iter)
                )
                for p in range(self._t_prof.n_seats)
            ]

        # reinitialize the network everytime, but you can choose to use 'last' init mode
        self._ray.wait([self._ray.remote(self._ps_handles[p_id].reset_avrg_net)])
        self._update_leaner_actors(update_avrg_for_plyrs=[p_id])

        SMOOTHING = 200
        accumulated_averaged_high_loss = 0.0
        accumulated_averaged_low_loss = 0.0

        if cfr_iter > 0:
            for epoch_nr in tqdm(range(self._avrg_args.n_batches_avrg_training)):
                t0 = time.time()

                # Compute gradients # TODO: train the high- and low-level components in different frequency
                high_grads_from_all_las, low_grads_from_all_las,\
                _averaged_high_loss, _averaged_low_loss = self._get_avrg_gradients(p_id=p_id)

                accumulated_averaged_high_loss += _averaged_high_loss
                accumulated_averaged_low_loss += _averaged_low_loss

                t_computation += time.time() - t0

                # Applying gradients to the network on PS
                t0 = time.time()
                self._ray.wait([
                    self._ray.remote(self._ps_handles[p_id].apply_grads_high_avrg, high_grads_from_all_las)
                ])
                self._ray.wait([
                    self._ray.remote(self._ps_handles[p_id].apply_grads_low_avrg, low_grads_from_all_las)
                ])

                # Step LR scheduler on PS
                self._ray.wait([
                    self._ray.remote(self._ps_handles[p_id].step_scheduler_avrg, _averaged_high_loss, _averaged_low_loss)
                ])

                # update AvrgStrategyNet on all las from PS
                self._update_leaner_actors(update_avrg_for_plyrs=[p_id])

                # log current loss
                if self._t_prof.log_verbose and ((epoch_nr + 1) % SMOOTHING == 0):
                    self._ray.wait([
                        self._ray.remote(self._chief_handle.add_scalar, exp_high_loss_each_p[p_id],
                                         "HDCFR_NN_Losses/Average_High", epoch_nr,
                                         accumulated_averaged_high_loss / SMOOTHING)
                    ])
                    accumulated_averaged_high_loss = 0.0
                    self._ray.wait(
                        [
                            self._ray.remote(self._chief_handle.add_scalar, exp_low_loss_each_p[p_id],
                                             "HDCFR_NN_Losses/Average_Low", epoch_nr,
                                             accumulated_averaged_low_loss / SMOOTHING
                                             )
                        ]
                    )
                    accumulated_averaged_low_loss = 0.0

                t_syncing += time.time() - t0

        return t_computation, t_syncing

    def _get_avrg_gradients(self, p_id):
        high_grads = [
            self._ray.remote(la.get_high_avrg_grads, p_id)
            for la in self._la_handles
        ]
        self._ray.wait(high_grads)

        low_grads = [
            self._ray.remote(la.get_low_avrg_grads, p_id)
            for la in self._la_handles
        ]
        self._ray.wait(low_grads)

        losses = self._ray.get([
            self._ray.remote(la.get_loss_last_batch_avrg, p_id)
            for la in self._la_handles
        ])

        high_losses = [loss[0] for loss in losses if loss is not None and loss[0]]
        low_losses = [loss[1] for loss in losses if loss is not None and loss[1]]

        averaged_high_loss = sum(high_losses) / float(len(high_losses)) if len(high_losses) > 0 else -1
        averaged_low_loss = sum(low_losses) / float(len(low_losses)) if len(low_losses) > 0 else -1

        return high_grads, low_grads, averaged_high_loss, averaged_low_loss

    # train adv net

    def _train_adv(self, p_id, cfr_iter):
        t_computation = 0.0
        t_syncing = 0.0

        # For logging the loss to see convergence in Tensorboard
        if self._t_prof.log_verbose:
            exp_high_loss_each_p = [
                self._ray.remote(
                    self._chief_handle.create_experiment,
                    self._t_prof.name + "_ADV_High_Loss_P" + str(p) + "_I" + str(cfr_iter)
                )
                for p in range(self._t_prof.n_seats)
            ]
            exp_low_loss_each_p = [
                self._ray.remote(
                    self._chief_handle.create_experiment,
                    self._t_prof.name + "_ADV_Low_Loss_P" + str(p) + "_I" + str(cfr_iter)
                )
                for p in range(self._t_prof.n_seats)
            ]

        if (cfr_iter % self._t_prof.periodic_restart == 0):
            self._ray.wait([
                self._ray.remote(self._ps_handles[p_id].reset_adv_net, cfr_iter)
            ])
            NB = self._adv_args.n_batches_adv_training
        else:
            NB = int(self._adv_args.n_batches_adv_training / 5)

        self._update_leaner_actors(update_adv_for_plyrs=[p_id])

        SMOOTHING = 200
        accumulated_averaged_high_loss = 0.0
        accumulated_averaged_low_loss = 0.0

        for epoch_nr in tqdm(range(NB)):
            t0 = time.time()

            # Compute gradients
            high_grads_from_all_las, low_grads_from_all_las,\
            _averaged_high_loss, _averaged_low_loss = self._get_adv_gradients(p_id=p_id)

            accumulated_averaged_high_loss += _averaged_high_loss
            accumulated_averaged_low_loss += _averaged_low_loss

            t_computation += time.time() - t0

            # Applying gradients
            t0 = time.time()
            self._ray.wait([
                self._ray.remote(self._ps_handles[p_id].apply_grads_high_adv, high_grads_from_all_las)
            ])
            self._ray.wait(
                [
                    self._ray.remote(self._ps_handles[p_id].apply_grads_low_adv, low_grads_from_all_las)
                ]
            )

            # Step LR scheduler
            self._ray.wait([
                self._ray.remote(self._ps_handles[p_id].step_scheduler_adv, _averaged_high_loss, _averaged_low_loss)
            ])

            # update ADV on all las
            self._update_leaner_actors(update_adv_for_plyrs=[p_id])

            # log current loss
            if self._t_prof.log_verbose and ((epoch_nr + 1) % SMOOTHING == 0):
                self._ray.wait([
                    self._ray.remote(self._chief_handle.add_scalar, exp_high_loss_each_p[p_id],
                                     "HDCFR_NN_Losses/Advantage_High", epoch_nr,
                                     accumulated_averaged_high_loss / SMOOTHING)
                ])
                accumulated_averaged_high_loss = 0.0
                self._ray.wait([
                    self._ray.remote(self._chief_handle.add_scalar, exp_low_loss_each_p[p_id],
                                     "HDCFR_NN_Losses/Advantage_Low", epoch_nr,
                                     accumulated_averaged_low_loss / SMOOTHING)
                ])
                accumulated_averaged_low_loss = 0.0

            t_syncing += time.time() - t0

        return t_computation, t_syncing

    def _get_adv_gradients(self, p_id):
        high_grads = [
            self._ray.remote(la.get_high_adv_grads, p_id)
            for la in self._la_handles
        ]
        self._ray.wait(high_grads)

        low_grads = [
            self._ray.remote(la.get_low_adv_grads, p_id)
            for la in self._la_handles
        ]
        self._ray.wait(low_grads)

        losses = self._ray.get([
            self._ray.remote(la.get_loss_last_batch_adv, p_id)
            for la in self._la_handles
        ])

        high_losses = [loss[0] for loss in losses if loss is not None and loss[0]]
        low_losses = [loss[1] for loss in losses if loss is not None and loss[1]]

        averaged_high_loss = sum(high_losses) / float(len(high_losses)) if len(high_losses) > 0 else -1
        averaged_low_loss = sum(low_losses) / float(len(low_losses)) if len(low_losses) > 0 else -1

        return high_grads, low_grads, averaged_high_loss, averaged_low_loss

    # train baseline net

    def _train_baseline(self, n_updates, cfr_iter):
        t_computation = 0.0
        t_syncing = 0.0

        if self._t_prof.log_verbose:
            exp_loss = [
                self._ray.remote(
                    self._chief_handle.create_experiment,
                    self._t_prof.name + "_Baseline_Loss" + "_I" + str(cfr_iter)
                )
            ]

        # self._ray.wait([
        #     self._ray.remote(self._ps_handles[0].reset_baseline_net, )
        # ])
        # TODO: reset the baseline net every training iteration
        self._ray.wait([self._ray.remote(self._ps_handles[0].reset_baseline_net)])
        self._update_leaner_actors(update_baseline=True)

        SMOOTHING = 200
        accumulated_averaged_loss = 0.0

        for epoch_nr in tqdm(range(n_updates)):
            t0 = time.time()

            # Compute gradients
            grads_from_all_las = [
                self._ray.remote(la.get_baseline_grads, )
                for la in self._la_handles
            ]
            self._ray.wait(grads_from_all_las)

            losses = self._ray.get([
                self._ray.remote(la.get_loss_last_batch_baseline, )
                for la in self._la_handles
            ])
            losses = [loss for loss in losses if loss is not None]
            n = len(losses)
            loss = sum(losses) / float(n) if n > 0 else -1
            accumulated_averaged_loss += loss

            t_computation += time.time() - t0

            # Applying gradients
            t0 = time.time()
            self._ray.wait([
                self._ray.remote(self._ps_handles[0].apply_grads_baseline, grads_from_all_las)
            ])

            # update Baseline on all las
            self._update_leaner_actors(update_baseline=True)

            if self._t_prof.log_verbose and ((epoch_nr + 1) % SMOOTHING == 0):
                self._ray.wait([
                    self._ray.remote(self._chief_handle.add_scalar, exp_loss[0],
                                     "HDCFR_NN_Losses/Baseline", epoch_nr, accumulated_averaged_loss / SMOOTHING)
                ])
                accumulated_averaged_loss = 0.0

            t_syncing += time.time() - t0

        return t_computation, t_syncing

    def _generate_traversals(self, p_id, cfr_iter):
        t_gen = time.time()
        states_seen = self._ray.get([
            self._ray.remote(la.generate_data, p_id, cfr_iter)
            for la in self._la_handles
        ]) # not in a sequential manner
        t_gen = time.time() - t_gen

        if p_id == 1:
            self._ray.wait([
                self._ray.remote(self._chief_handle.add_scalar, self._exp_states_traversed, "States Seen", cfr_iter,
                                 sum(states_seen))
            ])

        return t_gen

    def _generate_target_b(self, cfr_iter):
        t_gen = time.time()
        self._ray.get([
            self._ray.remote(la.get_target_b, cfr_iter)
            for la in self._la_handles
        ])
        t_gen = time.time() - t_gen

        return t_gen

    def _reset_baseline_buffer(self):
        self._ray.get(
            [
                self._ray.remote(la.reset_baseline_buffer, )
                for la in self._la_handles
            ]
        ) # TODO: check, use ray.wait
        # self._ray.wait(
        #     [
        #         self._ray.remote(la.reset_baseline_buffer, )
        #         for la in self._la_handles
        #     ]
        # )

    def _baseline_buffer_to_list(self):
        self._ray.get(
            [
                self._ray.remote(la.baseline_buffer_to_list, )
                for la in self._la_handles
            ]
        ) # TODO: check， use ray.wait
        # self._ray.wait(
        #     [
        #         self._ray.remote(la.baseline_buffer_to_list, )
        #         for la in self._la_handles
        #     ]
        # )

        # main logic

    def run_one_iter_alternating_update(self, cfr_iter):
        t_generating_data = 0.0
        t_computation_adv = 0.0
        t_syncing_adv = 0.0

        # reset the baseline buffer
        print("Reseting the baseline buffer ......")
        self._reset_baseline_buffer()

        # generate data # TODO: use alternative update like the other algorithms
        self._update_leaner_actors(update_adv_for_plyrs=self._all_p_aranged, update_baseline=True)
        print("Generating Data...")
        for p_learning in range(self._t_prof.n_seats):
            _t_generating_data = self._generate_traversals(p_id=p_learning, cfr_iter=cfr_iter)
            t_generating_data += _t_generating_data

        # train the adv net
        print("Training Advantage Net...")
        for p_learning in range(self._t_prof.n_seats):
            _t_computation_adv, _t_syncing_adv = self._train_adv(p_id=p_learning, cfr_iter=cfr_iter)
            t_computation_adv += _t_computation_adv
            t_syncing_adv += _t_syncing_adv

        # train the baseline net
        # self._update_leaner_actors(update_adv_for_plyrs=self._all_p_aranged)
        self._baseline_buffer_to_list()
        self._generate_target_b(cfr_iter=cfr_iter)
        t_computation_baseline, t_syncing_baseline = \
            self._train_baseline(n_updates=self._baseline_args.n_batches_per_iter_baseline, cfr_iter=cfr_iter)

        print("Synchronizing...")
        self._update_leaner_actors(update_adv_for_plyrs=self._all_p_aranged, update_baseline=True)

        ret = {
            "t_generating_data": t_generating_data,
            "t_computation_adv": t_computation_adv,
            "t_syncing_adv": t_syncing_adv,
            "t_computation_baseline": t_computation_baseline,
            "t_syncing_baseline": t_syncing_baseline
        }

        return ret

