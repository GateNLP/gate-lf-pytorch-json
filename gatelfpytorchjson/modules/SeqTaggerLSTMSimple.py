import torch.nn
from gatelfpytorchjson import CustomModule
from gatelfpytorchjson import EmbeddingsModule
import sys
import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)


class SeqTaggerLSTMSimple(CustomModule):

    def get_init_weights(self, batchsize):
        """A method that returns a tensor that can be used to initialize the hidden states of the LSTM.
        """
        # NOTE: this can be used in forward to use a weight initialization that is different from
        # the default zero initialization.

        # first get the parameters so we can create the same type of vector, from the same device easily
        # the lstm is really wrapped in a a "takefromtuple" layer, so we have to get it out of there
        lstmparms = next(self.layer_lstm.parameters()).data
        h0 = lstmparms.new(self.layer_lstm.num_layers, batchsize, self.layer_lstm.hidden_size)
        c0 = lstmparms.new(self.layer_lstm.num_layers, batchsize, self.layer_lstm.hidden_size)
        h0.copy_(torch.randn(self.layer_lstm.num_layers, batchsize, self.layer_lstm.hidden_size)*0.01)
        c0.copy_(torch.randn(self.layer_lstm.num_layers, batchsize, self.layer_lstm.hidden_size)*0.01)
        return h0, c0

    # TODO: make sure the dataset is not stored and only used to decide on parameters etc so
    # that the dataset data is not getting pickled when the model is saved!
    def __init__(self, dataset, config={}):
        super().__init__(config=config)
        torch.manual_seed(1)
        self.n_classes = dataset.get_info()["nClasses"]
        logger.debug("Initializing module SeqTaggerLSTMSimple for classes: %s" % (self.n_classes,))
        # create a simple LSTM-based network: the input uses an embedding layer created from
        # the vocabulary, then a bidirectional LSTM followed by a simple softmax layer for each element
        # in the sequence
        feature = dataset.get_index_features()[0]
        vocab = feature.vocab
        logger.debug("Initializing module SeqTaggerLSTMSimple for classes: %s and vocab %s" %
                     (self.n_classes, vocab, ))

        # create the embedding layer from the vocab
        self.layer_emb = EmbeddingsModule(vocab)
        emb_dims = self.layer_emb.emb_dims

        # NOTE: instead of using our own EmbeddingsModule, we could also just use
        # a fixed embedding layer here:
        # emb_dims = 100
        # emb_size = vocab.size()
        # self.layer_emb = torch.nn.Embedding(emb_size, emb_dims, padding_idx=0)

        # configure the LSTM; this can be overriden using config parameters
        # TODO: for now just fixed parms!
        self.lstm_hiddenunits = 200
        self.lstm_nlayers = 1
        self.lstm_is_bidirectional = False
        self.layer_lstm = torch.nn.LSTM(
            input_size=emb_dims,
            hidden_size=self.lstm_hiddenunits,
            num_layers=self.lstm_nlayers,
            dropout=0.0,
            bidirectional=self.lstm_is_bidirectional,
            batch_first=True)
        lin_units = self.lstm_hiddenunits*2 if self.lstm_is_bidirectional else self.lstm_hiddenunits
        self.lstm_totalunits = lin_units
        self.layer_lin = torch.nn.Linear(lin_units, self.n_classes)
        # Note: the log-softmax function is used directly in forward, we do not define a layer for that
        logger.info("Network created: %s" % (self, ))

    # this gets a batch if independent variables
    # By default this is in reshaped padded batch format.
    # For sequences and a single feature this has the format:
    # * a list containing a sublist for each instance
    # * each sublist contains one nested list with the padded sequence of word indices
    #   (by default the padding index is 0)
    # Note: the calling model wrapper does not automatically put the batch on cuda,
    # so if we want this, it has to be done explicitly in here, using the method
    # on_cuda() to check if cuda is enabled for the module.
    # HOWEVER: we have decided not to put the indices on cuda
    def forward(self, batch):
        # we need only the first feature:
        # print("DEBUG: batch=", batch, file=sys.stderr)
        batch = torch.LongTensor(batch[0])
        logger.debug("forward called with batch of size %s: %s" % (batch.size(), batch,))
        if self.on_cuda():
            batch.cuda()
        tmp_embs = self.layer_emb(batch)
        # hidden_init = self.get_init_weights(len(batch))
        lstm_hidden, (lstm_c_last, lstm_h_last) = self.layer_lstm(tmp_embs)

        tmp_lin = self.layer_lin(lstm_hidden)
        # out = self.layer_out(tmp_lin)
        out = F.log_softmax(tmp_lin, 2)
        logger.debug("output tensor is if size %s: %s" % (out.size(), out, ))
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

