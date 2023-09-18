# Copyright (c) 2019 Eric Steinberger


import ray, torch

from PokerRL.eval.rl_br.LocalRLBRMaster import LocalRLBRMaster as _LocalEvalRLBRMaster


@ray.remote(num_cpus=1, num_gpus=0.2 if torch.cuda.is_available() else 0)
class DistRLBRMaster(_LocalEvalRLBRMaster):

    def __init__(self, t_prof, chief_handle, eval_agent_cls):
        super().__init__(t_prof=t_prof, chief_handle=chief_handle, eval_agent_cls=eval_agent_cls)
