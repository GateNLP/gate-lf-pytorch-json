import torch.nn
from . CustomModule import CustomModule
from . embeddingsmodule import EmbeddingsModule
import sys
import logging
import torch.nn.functional as F

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


class LayerCNN(CustomModule):
    """
    LayerCNN handles a single input of shape (batchsize, maxseqlen, embdims)
    and creates everything to get a final output of hidden units
    (including batch normalization, dropout and non-linearity)
    The number of output units is in self.dim_outputs after initialisation.
    """
    def __init__(self, emb_dims, config={}, **kwargs):
        super(LayerCNN, self).__init__(config=config)
        logger.debug("Building LayerCNN module, config=%s" % (config, ))

        self.rand_seed = kwargs.get("seed") or 1
        self.emb_dims = emb_dims
        self.channels_out = kwargs.get("channels_out") or 100
        self.kernel_sizes = kwargs.get("kernel_sizes") or [3, 4, 5]
        self.dropout_prob = kwargs.get("dropout") or 0.6
        self.use_batchnorm = kwargs.get("use_batchnorm") or True
        nonlin = torch.nn.ReLU()


        # TODO This should get removed and the set_seed() method inherited should
        # get used instead
        torch.manual_seed(self.rand_seed)


        # Architecture:
        # for each kernel size we create a separate CNN
        # Note: batchnormalization will be applied before  the nonlinearity for now!

        layers_cnn = torch.nn.ModuleList()
        for K in self.kernel_sizes:
            layer_cnn = torch.nn.Sequential()
            layername = "conv1d_K{}".format(K)
            layer_cnn.add_module(layername,
                torch.nn.Conv1d(in_channels=self.emb_dims,
                                out_channels=self.channels_out,
                                kernel_size=K,
                                stride=1,
                                padding=int(K/2),
                                dilation=1,
                                groups=1,
                                bias=True)
            )
            if self.use_batchnorm:
                layername = "batchnorm1d_K{}".format(K)
                layer_cnn.add_module(layername, torch.nn.BatchNorm1d(self.channels_out))
            layer_cnn.add_module("nonlin_K{}".format(K), nonlin)
            layer_cnn.add_module("maxpool_K{}".format(K), MaxFrom1d(dim=-1))
            layer_cnn.add_module("dropout_K{}".format(K), torch.nn.Dropout(self.dropout_prob))
            layers_cnn.append(layer_cnn)

        # each convolution layer gives us channels_out outputs, and we have as many of
        # of those as we have kernel sizes
        self.dim_outputs = len(self.kernel_sizes)*self.channels_out

        self.layers = torch.nn.Sequential()
        self.layers.add_module("transpose", Transpose4CNN())
        self.layers.add_module("CNNs", ListModule(layers_cnn))
        self.layers.add_module("concat", Concat(dim=1))

        # Note: the log-softmax function is used directly in forward, we do not define a layer for that
        # logger.info("Layer created: %s" % (self, ))

    def forward(self, batch):
        # batch is assumed to already be a tensor of the correct shape
        # batchsize, maxseq, embdims
        if self.on_cuda():
            batch.cuda()
        out = self.layers(batch)
        # logger.debug("output tensor is if size %s: %s" % (out.size(), out, ))
        return out

