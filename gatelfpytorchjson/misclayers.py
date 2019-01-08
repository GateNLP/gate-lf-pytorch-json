import torch
import sys
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)


class MaxFrom1d(torch.nn.Module):
    """
    Simple maxpool module that takes the maximum from one dimension of a tensor and
    reduces the tensor dimensions by 1.
    Essentially the same as torch.max(x, dim=thedimension)
    """
    def __init__(self, dim=-1):
        super(MaxFrom1d, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.max(x, dim=self.dim)[0]


class Concat(torch.nn.Module):
    """
    Simple module that will concatenate a list of inputs across a dimension
    """
    def __init__(self, dim=-1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, listofx):
        return torch.cat(listofx, self.dim)


class Transpose4CNN(torch.nn.Module):
    """
    Does the transposing for CNN
    """
    def __init__(self):
        super(Transpose4CNN, self).__init__()

    def forward(self, x):
        return x.transpose(1,2)


class ListModule(torch.nn.Module):
    """
    Simple module that runs the same input through all modules in a modulelist
    and returns a list of outputs
    """
    def __init__(self, modulelist):
        super(ListModule, self).__init__()
        self.modulelist = modulelist

    def forward(self, x):
        return [l(x) for l in self.modulelist]

