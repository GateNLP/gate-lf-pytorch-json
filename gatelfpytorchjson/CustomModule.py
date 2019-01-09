import torch
import torch.nn
import sys
import logging
from abc import abstractmethod

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)


class CustomModule(torch.nn.Module):

    def __init__(self, config={}):
        super().__init__()
        # for caching the cuda status, is set when on_cuda() is called the first time
        seed = config.get("seed", None)
        if seed is not None:
            self.set_seed(seed)
        self._on_cuda = None

    def get_lossfunction(self, config={}):
        return torch.nn.NLLLoss(ignore_index=-1)

    def get_optimizer(self, config={}):
        parms = filter(lambda p: p.requires_grad, self.parameters())
        return torch.optim.Adam(parms, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        # make sure it is set on all GPUs as well, we can always do this as torch ignores
        # this if no CUDA is available
        torch.cuda.manual_seed_all(seed)

    def on_cuda(self):
        """Returns true or false depending on if the module is on cuda or not. Unfortunately
        there is no API method in PyTorch for this so we get this from the first parameter of the
        model and cache it.
        NOTE: this must be called outside of the init() method, because the cuda status of the module
        gets set by the modelwrapper.
        """
        if self._on_cuda is None:
            self._on_cuda = next(self.parameters()).is_cuda
        return self._on_cuda

    @abstractmethod
    def forward(self, *input):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_on_cuda"]
        return state

    def __setstate__(self, state):
        state["_on_cuda"] = None
        self.__dict__.update(state)
