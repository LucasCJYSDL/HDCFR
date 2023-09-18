# Copyright (c) Eric Steinberger 2020

import ray
import torch

from DREAM_and_DeepCFR.workers.ps.local import ParameterServer as _LocalParameterServer


# @ray.remote(num_cpus=0.2, num_gpus=0.2 if torch.cuda.is_available() else 0)
@ray.remote(num_cpus=1)
class ParameterServer(_LocalParameterServer):

    def __init__(self, t_prof, owner, chief_handle):
        super().__init__(t_prof=t_prof, owner=owner, chief_handle=chief_handle)