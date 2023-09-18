from PokerRL.rl.base_cls.workers.DriverBase import DriverBase

from HDCFR.EvalAgent import EvalAgent
from HDCFR.workers.driver._HighLevelAlgo import HighLevelAlgo

class Driver(DriverBase):

    def __init__(self, t_prof, eval_methods, n_iterations=None, iteration_to_import=None, name_to_import=None):
        if t_prof.DISTRIBUTED:
            from HDCFR.workers.chief.dist import Chief
            from HDCFR.workers.la.dist import LearnerActor
            from HDCFR.workers.ps.dist import ParameterServer

        else:
            from HDCFR.workers.chief.local import Chief
            from HDCFR.workers.la.local import LearnerActor
            from HDCFR.workers.ps.local import ParameterServer

        # the main thing to check when introducing more eval methods
        super().__init__(t_prof=t_prof, eval_methods=eval_methods, n_iterations=n_iterations,
                         iteration_to_import=iteration_to_import, name_to_import=name_to_import,
                         chief_cls=Chief, eval_agent_cls=EvalAgent)

        print("Creating LAs...")
        self.la_handles = [
            self._ray.create_worker(LearnerActor,
                                    t_prof,
                                    i,
                                    self.chief_handle)
            for i in range(t_prof.n_learner_actors)
        ]

        print("Creating Parameter Servers...")
        self.ps_handles = [
            self._ray.create_worker(ParameterServer,
                                    t_prof,
                                    p,
                                    self.chief_handle)
            for p in range(t_prof.n_seats)
        ]

        self._ray.wait([
            self._ray.remote(self.chief_handle.set_ps_handle, *self.ps_handles),
            self._ray.remote(self.chief_handle.set_la_handles, *self.la_handles)
        ])

        print("Created and initialized Workers")

        self.algo = HighLevelAlgo(t_prof=t_prof,
                                  la_handles=self.la_handles,
                                  ps_handles=self.ps_handles,
                                  chief_handle=self.chief_handle)

        assert EvalAgent.EVAL_MODE_AVRG_NET in self._t_prof.eval_modes_of_algo
        assert not EvalAgent.EVAL_MODE_SINGLE in self._t_prof.eval_modes_of_algo
        assert t_prof.sampler == "learned_baseline"

        self._maybe_load_checkpoint_init()

    def load_checkpoint(self, step, name_to_load):
        # Call on all other workers sequentially to be safe against RAM overload
        for w in self.la_handles + self.ps_handles + [self.chief_handle]: # only effective for ps_handles
            self._ray.wait([
                self._ray.remote(w.load_checkpoint, name_to_load, step)
            ])

    def run(self):
        print("Setting stuff up...")
        # """"""""""""""""
        # Init globally
        # """"""""""""""""
        self.algo.init()

        print("Starting Training...")
        for _iter_nr in range(self.n_iterations):
            print("Iteration: ", self._iteration)

            # periodically evaluate
            avrg_times = None
            if self._any_eval_needs_avrg_net():
                avrg_times = self.algo.train_average_nets(cfr_iter=_iter_nr)

            ## only run in a certain frequency
            self.evaluate()

            self.periodically_checkpoint()

            self.periodically_export_eval_agent()

            # train the adv net and baseline net
            iter_times = self.algo.run_one_iter_alternating_update(cfr_iter=self._iteration)

            print(
                "Generating Data: ", str(iter_times["t_generating_data"]) + "s.",
                "  ||  Trained ADV", str(iter_times["t_computation_adv"]) + "s.",
                "  ||  Synced ADV", str(iter_times["t_syncing_adv"]) + "s.",
                "\n",
                "Trained Baseline", str(iter_times["t_computation_baseline"]) + "s.",
                "  ||  Synced Baseline", str(iter_times["t_syncing_baseline"]) + "s."
            )

            if avrg_times:
                print(
                    "Trained AVRG", str(avrg_times["t_computation_avrg"]) + "s.",
                    "  ||  Synced AVRG", str(avrg_times["t_syncing_avrg"]) + "s.",
                )

            if self._iteration % self._t_prof.log_export_freq == 0:
                self.save_logs()

            self._iteration += 1




    def evaluate(self):
        """
        puts whole network on wait while the parameters are synced, but evaluates while training the next iteration(s)
        """
        evaluators_to_run = []
        for kind in list(self.eval_masters.keys()):
            ev = self.eval_masters[kind][0]
            freq = self.eval_masters[kind][1]
            if self._iteration % freq == 0:
                evaluators_to_run.append((ev, kind))
                print("Evaluating vs.", kind.upper())
                self._ray.wait([
                    self._ray.remote(ev.update_weights)
                ])

        for ev, kind in evaluators_to_run:
            if kind == "br":
                self._ray.remote(ev.evaluate, self._iteration, True)
            else:
                self._ray.remote(ev.evaluate, self._iteration)

    def _any_eval_needs_avrg_net(self):
        for e in list(self.eval_masters.values()):
            if self._iteration % e[1] == 0:
                return True
        return False

    def checkpoint(self, **kwargs):
        # Call on all other workers sequentially to be safe against RAM overload
        for w in self.la_handles + self.ps_handles + [self.chief_handle]: # only work for ps
            self._ray.wait([
                self._ray.remote(w.checkpoint, self._iteration)
            ])

        # Delete past checkpoints
        s = [self._iteration]
        if self._iteration > self._t_prof.checkpoint_freq + 1:
            s.append(self._iteration - self._t_prof.checkpoint_freq) # keep the current one and the previous one

        # self._delete_past_checkpoints(steps_not_to_delete=s)

