import torch.nn
from gatelfpytorchjson.CustomModule import CustomModule
import sys
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)


class LayerLSTM(CustomModule):
    """
    LayerCNN handles a single input of shape (batchsize, maxseqlen, embdims)
    and creates everything to get a final output of hidden units
    (including batch normalization, dropout and non-linearity)
    The number of output units is in self.dim_outputs after initialisation.
    """
    def __init__(self, emb_dims, config={}):
        super(LayerLSTM, self).__init__(config=config)
        logger.debug("Building LayerLSTM module, config=%s" % (config, ))

        self.emb_dims = emb_dims
        self.channels_out = config.get("channels_out", 100)
        self.batch_first = config.get("batch_first", True)
        self.bidirectional = config.get("bidirectional", True)
        self.dropout_prob = config.get("dropout", 0.6)
        self.use_batchnorm = config.get("use_batchnorm", True)

        self.lstm = torch.nn.LSTM(self.emb_dims, self.channels_out, batch_first=self.batch_first,
                                  bidirectional=self.bidirectional)

        # Note: the log-softmax function is used directly in forward, we do not define a layer for that
        # logger.info("Layer created: %s" % (self, ))
        if self.bidirectional:
            self.out_dims = self.channels_out*2
        else:
            self.out_dims = self.channels_out

    def forward(self, batch):
        # batch is assumed to already be a tensor of the correct shape
        # batchsize, maxseq, embdims
        if self.on_cuda():
            batch.cuda()
        lstmed, hidden = self.lstm(batch)
        # logger.debug("output tensor is if size %s: %s" % (out.size(), out, ))
        return lstmed

