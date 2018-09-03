import torch.nn
from gatelfpytorchjson import CustomModule
from gatelfpytorchjson import EmbeddingsModule
import sys
import logging
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)


class SentClassCNN(CustomModule):

    def __init__(self, dataset, config={}):
        # !!! See our working notes!!!!
        super().__init__(config=config)
        logger.debug("Building SentClassCNN network, config=%s" % (config, ))
        # if we should use packed sequences or not for the LSTM
        self.use_packed = True

        # TODO This should get removed and the set_seed() method inherited should
        # get used instead
        torch.manual_seed(1)
        self.n_classes = dataset.get_info()["nClasses"]
        logger.debug("Initializing module SentClassCNN for classes: %s" % (self.n_classes,))

        feature = dataset.get_indexlist_features()[0]
        vocab = feature.vocab
        logger.debug("Initializing module SentClassCNN for classes: %s and vocab %s" %
                     (self.n_classes, vocab, ))
        # these layers replace the NgramModule default
        self.layer_emb = EmbeddingsModule(vocab)
        emb_dims = self.layer_emb.emb_dims

        # Architecture:
        # * have k conv layers, for kernel sizes ksize=3,4,5 and features nfeatures= 100,100,100
        # * input is a single embedding with random initialization channel for now
        # * (to replicate kim we would need to combine several different channels ...)

        n_features = 100
        self.dropout_prob = 0.2
        self.layer_cnn_k3 = torch.nn.Conv1d(
                in_channels=emb_dims,
                out_channels=n_features,
                kernel_size=3
            )
        self.layer_cnn_k4 = torch.nn.Conv1d(
                in_channels=emb_dims,
                out_channels=n_features,
                kernel_size=4
            )
        self.layer_cnn_k5 = torch.nn.Conv1d(
                in_channels=emb_dims,
                out_channels=100,
                kernel_size=4
            )
        # NOTE: we could use a MaxPool1d layer for each convolution layer,
        # but the kernel size would need to match the output length that
        # we get, depending on the kernel size. So we would have to have
        # a different maxpooling layer for each convolution layer.
        # Instead we use the max_pool1d function in forward and choose
        # the kernel dynamically!
        # We also use the relu function instead of a layer
        # we also use the dropout function isntead of a Dropout layer
        self.nonlin = F.relu  # or leaky_relu or whatever

        # each convolution layer gives us n_features and we have 3 such layers,
        # so eventually we will get, for each sequence, 3*n_features values
        self.lin_inputs = 3*n_features
        self.layer_lin = torch.nn.Linear(self.lin_inputs, self.n_classes)
        # Note: the log-softmax function is used directly in forward, we do not define a layer for that
        logger.info("Network created: %s" % (self, ))

    def through_cnn(self, embs, cnn_layer):
        """Run through the CNN, ReLU, Maxpooling and then squeeze. Expects embs to be already transposed."""
        tmp = cnn_layer(embs)
        return F.dropout(F.max_pool1d(self.nonlin(tmp), tmp.size()[-1]).squeeze(2), self.dropout_prob)

    def forward(self, batch):
        # we need only the first feature:
        # print("DEBUG: batch=", batch, file=sys.stderr)
        batch = torch.LongTensor(batch[0])
        batchsize = batch.size()[0]

        # logger.debug("forward called with batch of size %s: %s" % (batch.size(), batch,))
        if self.on_cuda():
            batch.cuda()
        tmp_embs = self.layer_emb(batch)

        # for the convolution layers we need the exchange the length and "channel" dimensions
        tmp_embs_t = tmp_embs.transpose(1,2)

        # run the embeddings tensor through each of the convolution layers, then through a relu
        # then through a maxpooling then squeeze the last dimension.
        c1 = self.through_cnn(tmp_embs_t, self.layer_cnn_k3)
        c2 = self.through_cnn(tmp_embs_t, self.layer_cnn_k4)
        c3 = self.through_cnn(tmp_embs_t, self.layer_cnn_k5)
        tmp_conv = torch.cat([c1,c2,c3],1)

        tmp_lin = self.layer_lin(tmp_conv)
        # out = self.layer_out(tmp_lin)
        out = F.log_softmax(tmp_lin, 1)
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
