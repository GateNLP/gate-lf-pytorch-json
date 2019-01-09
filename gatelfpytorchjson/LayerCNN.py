import torch.nn
from gatelfpytorchjson.CustomModule import CustomModule
from gatelfpytorchjson.misclayers import  MaxFrom1d, Concat, Transpose4CNN, ListModule
import sys
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)


class LayerCNN(CustomModule):
    """
    LayerCNN handles a single input of shape (batchsize, maxseqlen, embdims)
    and creates everything to get a final output of hidden units
    (including batch normalization, dropout and non-linearity)
    The number of output units is in self.dim_outputs after initialisation.
    """
    def __init__(self, emb_dims, config={}):
        super(LayerCNN, self).__init__(config=config)
        logger.debug("Building LayerCNN module, config=%s" % (config, ))

        self.emb_dims = emb_dims
        self.channels_out = config.get("channels_out", 100)
        self.kernel_sizes = config.get("kernel_sizes", [3, 4, 5])
        if  isinstance(self.kernel_sizes, str):
            self.kernel_sizes = [int(x) for x in self.kernel_sizes.split(",")]
        self.dropout_prob = config.get("dropout", 0.6)
        self.use_batchnorm = config.get("use_batchnorm", True)
        nonlin_name = config.get("nonlin", "ReLU")
        if nonlin_name == "ReLU":
            nonlin = torch.nn.ReLU()
        elif nonlin_name == "ELU":
            nonlin = torch.nn.ELU()
        elif nonlin_name == "Tanh":
            nonlin = torch.nn.Tanh()

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

