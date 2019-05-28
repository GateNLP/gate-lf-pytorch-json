import torch.nn
from gatelfpytorchjson import CustomModule
from gatelfpytorchjson import LayerCNN
import sys
import logging
import os
from allennlp.modules.elmo import Elmo

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)

# TODO: !!! Check if and how we can switch the embedded Elmo module to evaluation mode so that
# we avoid keeping around gradients and maybe avoid other stuff!

class TextClassCnnSingleElmo(CustomModule):

    def __init__(self, dataset, config={}):
        super().__init__(config=config)
        logger.debug("Building single feature TextClassCnnSingle network, config=%s" % (config, ))

        # First get the parameters dictated by the data.
        # NOTE/TODO: eventually this should be done outside the module and config parameters!
        self.n_classes = dataset.get_info()["nClasses"]
        # For now, this modules always uses one feature, the first one if there are several
        feature = dataset.get_indexlist_features()[0]
        vocab = feature.vocab
        logger.debug("Initializing module TextClassCnnSingle for classes: %s and vocab %s" %
                     (self.n_classes, vocab, ))

        # create the layers: input embeddings layer, ngrammodule for the CNN, linear output and logsoftmax

        # self.layer_emb = EmbeddingsModule(vocab)
        self.maxSentLen = 500
        elmo_path = config['elmo']
        elmo_option_file = os.path.join(elmo_path, "elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json")
        elmo_weight_file = os.path.join(elmo_path, "elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5")
        # self.elmo = ElmoEmbedder(options_file=elmo_option_file, weight_file=elmo_weight_file)
        self.elmo = Elmo(elmo_option_file, elmo_weight_file, 2)

        config["ngram_layer"] = "cnn"
        config["dropout"] = 0.6
        config["channels_out"] = 100
        config["kernel_sizes"] = "3,4,5"  # or [3, 4, 5, 6]
        config["use_batchnorm"] = True
        config["nonlin"] = "ReLU"  # or ELU or Tanh
        layer_cnns = LayerCNN(1024*2, config=config)
        layer_lin = torch.nn.Linear(layer_cnns.dim_outputs, self.n_classes)
        logsoftmax = torch.nn.LogSoftmax(dim=1)

        self.layers = torch.nn.Sequential()
        self.layers.add_module("layer_cnns", layer_cnns)
        self.layers.add_module("linear", layer_lin)
        self.layers.add_module("logsoftmax", logsoftmax)

        # Note: the log-softmax function is used directly in forward, we do not define a layer for that
        logger.info("Network created: %s" % (self, ))

    def forward(self, batch):
        # we need only the first feature:
        # print("DEBUG: batch=", batch, file=sys.stderr)
        # print(batch)
        batch = torch.LongTensor(batch)
        # print(batch.shape)
        # batchsize = batch.size()[0]
        sent_len = batch.size()[1]
        # print(sent_len)
        if sent_len > self.maxSentLen:
            batch = batch[:, :self.maxSentLen, :]
        # print(batch.shape)

        # logger.debug("forward called with batch of size %s: %s" % (batch.size(), batch,))
        if self.on_cuda():
            batch = batch.type(torch.cuda.LongTensor)
            batch.cuda()
        # print(len(batch))
        # print(len(batch[0]))
        elmo_embed = self.elmo(batch)['elmo_representations']

        # print(len(elmo_embed))
        # print(elmo_embed[0].shape)
        # elmo_embed = torch.Tensor(elmo_embed)
        # avg_embed = elmo_embed[0]
        # embed = self.elmo.embed_sentences(batch[0])
        # avg_embd = [torch.mean(x, axis=0) for x in elmo_embed]
        avg_embd = torch.cat(elmo_embed, 2)
        # print(avg_embd.shape)
        # avg_embed = torch.FloatTensor(avg_embd)
        # if self.on_cuda():
        #     avg_embed.cuda()
    
        # if self.on_cuda():
        #     embed.cuda()
        out = self.layers(avg_embd)

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
