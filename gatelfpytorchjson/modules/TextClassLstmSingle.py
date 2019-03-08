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


class TextClassLstmSingle(CustomModule):

    def __init__(self, dataset, config={}):
        super().__init__(config=config)
        logger.debug("Building TextClassLstmSingle network, config=%s" % (config, ))
        # if we should use packed sequences or not for the LSTM
        self.use_packed = True

        # TODO This should get removed and the set_seed() method inherited should
        # get used instead
        torch.manual_seed(1)
        self.n_classes = dataset.get_info()["nClasses"]
        logger.debug("Initializing module TextClassLstmSingle for classes: %s" % (self.n_classes,))

        feature = dataset.get_indexlist_features()[0]
        vocab = feature.vocab
        logger.debug("Initializing module TextClassLstmSingle for classes: %s and vocab %s" %
                     (self.n_classes, vocab, ))
        # these layers replace the NgramModule default
        self.layer_emb = EmbeddingsModule(vocab)
        emb_dims = self.layer_emb.emb_dims

        self.lstm_hiddenunits = 50
        self.lstm_nlayers = 3
        self.lstm_is_bidirectional = True
        self.layer_lstm = torch.nn.LSTM(
            input_size=emb_dims,
            hidden_size=self.lstm_hiddenunits,
            num_layers=self.lstm_nlayers,
            dropout=0.0,
            bidirectional=self.lstm_is_bidirectional,
            batch_first=True)
        lin_units = self.lstm_hiddenunits
        if self.lstm_is_bidirectional:
            lin_units = lin_units * 2
        lin_units = lin_units * self.lstm_nlayers
        self.lstm_totalunits = lin_units
        logger.debug("Created LSTM input=%s, hidden=%s, nlayers=%s, bidir=%s, h_t_out=%s" %
          (emb_dims, self.lstm_hiddenunits, self.lstm_nlayers, self.lstm_is_bidirectional, lin_units))
        self.layer_lin = torch.nn.Linear(lin_units, self.n_classes)
        # Note: the log-softmax function is used directly in forward, we do not define a layer for that
        logger.info("Network created: %s" % (self, ))

    def forward(self, batch):
        # we need only the first feature:
        # print("DEBUG: batch=", batch, file=sys.stderr)
        batch = torch.LongTensor(batch[0])
        batchsize = batch.size()[0]

        # before we move this to CUDA, calculate masks and lengths
        mask = (batch != 0)  # 0 is the pad index, mask indicates non-padding, true value
        lengths = mask.sum(dim=1)

        # logger.debug("forward called with batch of size %s: %s" % (batch.size(), batch,))
        if self.on_cuda():
            batch.cuda()
        tmp_embs = self.layer_emb(batch)

        if not self.use_packed:
            # EITHER: if we want to use the padded sequences directly
            lstm_hidden, (lstm_h_last_orig, lstm_c_last_orig) = self.layer_lstm(tmp_embs)
            # Note: the lstm_h tensor is of shape nlayers*2ifbidir, batchsize, hidden,
            # and we want a tensor of shape batchsize, nlayers*2ifbidir*hidden.
            # Sadly, this cannot be achieved without copying the tensor data, but this should work:
            # - transpose(0,1): exchange first two dims, this gives batchsize, nlayers*2ifbidir, hidden
            # - contiguous(): make the transposed tensor contiguous so we can have a view
            # - view(batchsize,-1): merge the (now) last two dims for each batch instance into one dim
            lstm_h_last = lstm_h_last_orig.transpose(0,1).contiguous().view(batchsize,-1)
            # NOTE: the size of the second dimension should be equal to what we have defined for the linear
            # layer, i.e. self.lstm_totalunits
        else:
            # OR: if we want to pack the sequences:
            # before we pack the sequences, we have to sort them descending by length
            # for this, first sort the lengths tensor
            lengths_sorted, lengths_perm = lengths.sort(0, descending=True)
            # sort the actual data in the same way
            tmp_embs_sorted = tmp_embs[lengths_perm]
            packed = pack_padded_sequence(tmp_embs_sorted, lengths_sorted, batch_first=True)
            # logger.debug("Packed sequences %s" % (packed,))
            lstm_hidden_packed, (lstm_h_last_sorted, lstm_c_last_sorted) = self.layer_lstm(packed)
            _,unsort_perm = lengths_perm.sort(0)
            # Should we ever need the unsorted, padded outputs as well:
            # lstm_hidden_sorted = pad_packed_sequence(lstm_hidden_packed, batch_first=True, padding_value=0.0)
            # lstm_hidden = lstm_hidden_sorted[unsort_perm]
            # NOTE: the lstm_h_last_sorted is of shape numlayers*numdirections, batchsize, hiddensize
            # so we have to compress the first two dimensions into one
            # logger.debug("Last hidden: size=%s: %s" % (lstm_h_last_sorted.size(), lstm_h_last_sorted, ))
            lstm_h_last = lstm_h_last_sorted.transpose(0,1).contiguous().view(batchsize,-1)
            lstm_h_last = lstm_h_last[unsort_perm]
            # logger.debug("Shape of lstm_h_last=%s, orig=%s" % (lstm_h_last.size(), lstm_h_last_sorted.size() ))

        tmp_lin = self.layer_lin(lstm_h_last)
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
