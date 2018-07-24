import torch.nn
from torch.autograd import Variable as V
import torch.nn.functional as F
import sys

class CustomModule(torch.nn.Module):

    def __init__(self, config={}):
        super().__init__()
        # for caching the cuda status, is set when on_cuda() is called the first time
        self._on_cuda = None


    def get_lossfunction(self, config={}):
        return torch.nn.NLLLoss(ignore_index=-1)

    def get_optimizer(self, config={}):
        parms = filter(lambda p: p.requires_grad, self.parameters())
        return torch.optim.Adam(parms, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

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
