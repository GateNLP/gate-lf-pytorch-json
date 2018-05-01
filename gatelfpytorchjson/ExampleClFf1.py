import torch.nn
from torch.autograd import Variable as V
import torch.nn.functional as F
import sys


class ExampleClFf1(torch.nn.Module):

    # TODO: need to find out which parameters the constructor should get
    # Ideally, we should be completely agnostic of anything specific to the Dataset or meta information
    # here.
    # TODO: this should inherit from a class which already provides default implementations for
    # returning a loss function and for returning an optimizer
    @staticmethod
    def get_lossfunction(config={}):
        return torch.nn.NLLLoss(ignore_index=-1)

    @staticmethod
    def get_optimizer(parms, config={}):
        return torch.optim.Adam(parms, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    def __init__(self):
        super().__init__()
        self.fflayers = torch.nn.Sequential(
            torch.nn.Linear(34, 2, bias=True),
            torch.nn.Sigmoid(),
            torch.nn.LogSoftmax(dim=1)
        )
        self._on_cuda = None

    def on_cuda(self):
        """Returns true or false depending on if the module is on cuda or not. Unfortunately
        there is no API method in PyTorch for this so we get this from the first parameter of the
        model and cache it."""
        if self._on_cuda is None:
            self._on_cuda = next(self.parameters()).is_cuda
        return self._on_cuda

    def forward(self, batch):
        # batch is a list of features, each feature with all the values for all the batch examples
        # TODO: Check how to best deal with the shape mismatch and why we need to transpose here
        vals4pt = V(torch.FloatTensor(batch).t(), requires_grad=True)
        if self.on_cuda():
            vals4pt = vals4pt.cuda()
        out = self.fflayers(vals4pt)
        return out
