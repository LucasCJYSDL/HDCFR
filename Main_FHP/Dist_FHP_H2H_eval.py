import ray, torch
from FHP_H2H_eval import LocalH2HEvalMaster

@ray.remote(num_cpus=1, num_gpus=0.2 if torch.cuda.is_available() else 0)
class DistH2HEvalMaster(LocalH2HEvalMaster):
    def __init__(self, t_prof, chief_handle, eval_agent_cls):
        super().__init__(t_prof, chief_handle, eval_agent_cls)