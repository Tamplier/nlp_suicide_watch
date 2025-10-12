import gc
from contextlib import contextmanager
import torch

class GPUManager:
    def device(self):
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        else:
            return 'cpu'

    @classmethod
    @contextmanager
    def gpu_routine(cls, enter_gpu=None, exit_gpu=None):
        if torch.cuda.is_available():
            if enter_gpu:
                enter_gpu()
            with torch.no_grad():
                yield
            if exit_gpu:
                exit_gpu()
            gc.collect()
            torch.cuda.empty_cache()
        else:
            yield
