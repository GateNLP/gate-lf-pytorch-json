import torch.nn
from gatelfpytorchjson import CustomModule
from torch.autograd import Variable as V
import torch.nn.functional as F
import sys


class ExampleClFf1(CustomModule):

    # NOTE: make sure the dataset is not stored and only used to decide on parameters etc so
    # that the dataset data is not getting pickled when the model is saved!
    def __init__(self, dataset, config={}):
        super().__init__()
        self.fflayers = torch.nn.Sequential(
            torch.nn.Linear(34, 2, bias=True),
            torch.nn.Sigmoid(),
            torch.nn.LogSoftmax(dim=1)
        )
        self._on_cuda = None

    def forward(self, batch):
        # batch is a list of features, each feature with all the values for all the batch examples
        # TODO: Check how to best deal with the shape mismatch and why we need to transpose here
        vals4pt = V(torch.FloatTensor(batch).t(), requires_grad=True)
        if self.on_cuda():
            vals4pt = vals4pt.cuda()
        out = self.fflayers(vals4pt)
        return out