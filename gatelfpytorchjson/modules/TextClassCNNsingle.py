import torch.nn
from gatelfpytorchjson import CustomModule
from gatelfpytorchjson import EmbeddingsModule
from gatelfpytorchjson import NgramModule
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


class TextClassCNNsingle(CustomModule):

    def __init__(self, dataset, config={}):
        super().__init__(config=config)
        logger.debug("Building single feature TextClassCNNsingle network, config=%s" % (config, ))

        # First get the parameters dictated by the data.
        # NOTE/TODO: eventually this should be done outside the module and config parameters!
        self.n_classes = dataset.get_info()["nClasses"]
        # For now, this modules always uses one feature, the first one if there are several
        feature = dataset.get_indexlist_features()[0]
        vocab = feature.vocab
        logger.debug("Initializing module TextClassCNNsingle for classes: %s and vocab %s" %
                     (self.n_classes, vocab, ))

        # create the layers: input embeddings layer, ngrammodule for the CNN, linear output and logsoftmax

        layer_emb = EmbeddingsModule(vocab)
        config["ngram_layer"] = "cnn"
        config["dropout"] = 0.6
        config["channels_out"] = 100
        config["kernel_sizes"] = "3,4,5,6,7"  # or [3, 4, 5, 6]
        config["use_batchnorm"] = True
        config["nonlin"] = "ReLU"  # or ELU or Tanh
        ngrammodule = NgramModule(layer_emb, config=config)
        layer_lin = torch.nn.Linear(ngrammodule.out_dim, self.n_classes)
        logsoftmax = torch.nn.LogSoftmax(dim=1)

        self.layers = torch.nn.Sequential()
        self.layers.add_module("ngram-cnn", ngrammodule)
        self.layers.add_module("linear", layer_lin)
        self.layers.add_module("logsoftmax", logsoftmax)

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
