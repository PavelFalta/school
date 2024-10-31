from RedisQ.Base.Qlist import LW, LQ
from RedisQ.Base.Qjob import PredictionJob, PredictionContent
from RedisQ.BaseWorker import BaseWorker
import time
import requests
# import GPUtil as gpu


class GPUWorker(BaseWorker):
    """docstring for WSIWorker"""

    def get_availability(self):
        # return gpu.getAvailability(gpus, maxLoad=self.config.gpu_load,
                                            # maxMemory=self.config.gpu_memory)
        return True