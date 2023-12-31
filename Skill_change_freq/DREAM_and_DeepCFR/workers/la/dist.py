# Copyright (c) Eric Steinberger 2020

import ray
import torch

from DREAM_and_DeepCFR.workers.la.local import LearnerActor as LocalLearnerActor


# @ray.remote(num_cpus=0.2, num_gpus=0.2 if torch.cuda.is_available() else 0)
@ray.remote(num_cpus=1)
class LearnerActor(LocalLearnerActor):

    def __init__(self, t_prof, worker_id, chief_handle):
        LocalLearnerActor.__init__(self, t_prof=t_prof, worker_id=worker_id, chief_handle=chief_handle)
