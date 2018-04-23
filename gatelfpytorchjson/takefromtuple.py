import torch.nn


class TakeFromTuple(torch.nn.Module):

    def __init__(self, moduletowrap, which=0):
        """Wrap the model (e.g. LSTM) and make sure that only the which part of the
        tuple it creates is returned.
        """
        super().__init__()
        self.moduletowrap = moduletowrap

    def forward(self, vals):
        ret = self.moduletowrap(vals)
        if isinstance(ret, tuple):
            return ret[0]
        else:
            return ret
