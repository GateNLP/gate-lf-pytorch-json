import torch.nn
from gatelfpytorchjson import CustomModule
from gatelfpytorchjson import EmbeddingsModule
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


class SentClassCNN(CustomModule):

    def __init__(self, dataset, config={}, **kwargs):
        super(SentClassCNN, self).__init__(config=config)
        logger.debug("Building SentClassCNN network, config=%s" % (config, ))

        # TODO This should get removed and the set_seed() method inherited should
        # get used instead
        torch.manual_seed(1)

        # First get the parameters dictated by the data.
        # NOTE/TODO: eventually this should be done outside the module and config parameters!
        self.n_classes = dataset.get_info()["nClasses"]
        # For now, this modules always uses one feature, the first one if there are several
        feature = dataset.get_indexlist_features()[0]
        vocab = feature.vocab
        logger.debug("Initializing module SentClassCNN for classes: %s and vocab %s" %
                     (self.n_classes, vocab, ))
        # If we want to factor this in a separate CNNLayer module, the input should
        # probably already be proper embedding tensors, so this would need to get moved out
        layer_emb = EmbeddingsModule(vocab)
        self.emb_dims = layer_emb.emb_dims

        # other parameters, not dictated by the dataset but the defaults could
        # be adapted to the dataset. For now we used fixed defaults, similar to
        # what was used in the Kim paper
        self.channels_out = 100
        self.kernel_sizes = [3, 4, 5, 6, 7]
        self.dropout_prob = 0.5
        self.use_batchnorm = True
        nonlin = torch.nn.ReLU()

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
        self.lin_inputs = len(self.kernel_sizes)*self.channels_out
        layer_lin = torch.nn.Linear(self.lin_inputs, self.n_classes)

        self.layers = torch.nn.Sequential()
        self.layers.add_module("embs", layer_emb)
        self.layers.add_module("transpose", Transpose4CNN())
        self.layers.add_module("CNNs", ListModule(layers_cnn))
        self.layers.add_module("concat", Concat(dim=1))
        self.layers.add_module("linear", layer_lin)
        self.layers.add_module("logsoftmax", torch.nn.LogSoftmax(dim=1))

        # Note: the log-softmax function is used directly in forward, we do not define a layer for that
        logger.info("Network created: %s" % (self, ))

    def forward(self, batch):
        # we need only the first feature:
        # print("DEBUG: batch=", batch, file=sys.stderr)
        batch = torch.LongTensor(batch[0])
        batchsize = batch.size()[0]

        # logger.debug("forward called with batch of size %s: %s" % (batch.size(), batch,))
        if self.on_cuda():
            batch.cuda()
        out = self.layers(batch)
        # logger.debug("output tensor is if size %s: %s" % (out.size(), out, ))
        return out

    def get_lossfunction(self, config={}):
        # IMPORTANT: for the target indices, we use -1 for padding by default!
        return torch.nn.NLLLoss(ignore_index=-1)

    def get_optimizer(self, config={}):
        parms = filter(lambda p: p.requires_grad, self.parameters())
        # optimizer = torch.optim.SGD(parms, lr=0.01, momentum=0.9)
        # optimizer = torch.optim.SGD(parms, lr=0.01, momentum=0.9, weight_decay=0.05)
        optimizer = torch.optim.Adam(parms, lr=0.015, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        return optimizer
