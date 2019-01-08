import torch.nn
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)


class ConcatModules(torch.nn.Module):

    def __init__(self, inputs, name="ConcatModules", dim=None, config={}):
        """Concatenates the outputs of the given layers to a single output.
        The default axis for concatenating is the last dimension of the tensor.
        This can be overriden by setting dim to the axis.
        """
        super().__init__()
        self.inputs = inputs
        for i, input in enumerate(inputs):
            self.add_module(name+("-%s" % i), input)
        self.name = name
        self.dim = dim

    def forward(self, inputslist):
        if len(inputslist) != len(self.inputs):
            raise Exception("Number of modules and number of inputs differ")
        outs = []
        for i in range(len(inputslist)):
            out = self.inputs[i](inputslist[i])
            outs.append(out)
        axis = self.dim
        if not axis:
            axis = len(inputslist[0].size()-1)
        return torch.cat(outs, axis)

