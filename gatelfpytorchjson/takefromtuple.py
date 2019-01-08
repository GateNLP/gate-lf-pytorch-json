import torch.nn
import sys
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)


class TakeFromTuple(torch.nn.Module):

    def __init__(self, moduletowrap, which=0):
        """Wrap the model (e.g. LSTM) and make sure that only the which part of the
        tuple it creates is returned.
        """
        super().__init__()
        self.module = moduletowrap

    def forward(self, vals):
        ret = self.module(vals)
        if isinstance(ret, tuple):
            return ret[0]
        else:
            return ret
