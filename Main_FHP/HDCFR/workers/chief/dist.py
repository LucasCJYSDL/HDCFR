import ray

from HDCFR.workers.chief.local import Chief as _LocalChief


@ray.remote(num_cpus=1) # num_cpus can be fraction numbers
class Chief(_LocalChief):

    def __init__(self, t_prof):
        _LocalChief.__init__(self, t_prof=t_prof)
        # print(self._log_buf)