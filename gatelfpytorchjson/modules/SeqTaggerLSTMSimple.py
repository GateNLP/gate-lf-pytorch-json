import torch.nn
from gatelfpytorchjson import CustomModule
from gatelfpytorchjson import EmbeddingsModule
from gatelfpytorchjson import TakeFromTuple
from torch.autograd import Variable as V
import torch.nn.functional as F
from collections import OrderedDict
import sys


class SeqTaggerLSTMSimple(CustomModule):

    def get_init_weights(self, batchsize):
        """A method that returns a tensor that can be used to initialize the hidden states of the LSTM.
        """
        # first get the parameters so we can create the same type of vector, from the same device easily
        # the lstm is really wrapped in a a "takefromtuple" layer, so we have to get it out of there
        lstmlayer = dict(self.layers.named_modules())['lstm'].module
        lstmparms = next(lstmlayer.parameters()).data
        h1 = V(lstmparms.new(lstmlayer.num_layers, batchsize, lstmlayer.hidden_size).zero_())
        h2 = V(lstmparms.new(lstmlayer.num_layers, batchsize, lstmlayer.hidden_size).zero_())
        return (h1, h2)


    # NOTE: make sure the dataset is not stored and only used to decide on parameters etc so
    # that the dataset data is not getting pickled when the model is saved!
    def __init__(self, dataset, config={}):
        super().__init__(config=config)

        n_classes = dataset.get_info()["nClasses"]
        # create a simple LSTM-based network: the input uses an embedding layer created from
        # the vocabulary, then a bidirectional LSTM followed by a simple softmax layer for each element
        # in the sequence
        feature = dataset.get_index_features()[0]
        vocab = feature.vocab
        # create the embedding layer from the vocab
        emblayer = EmbeddingsModule(vocab)

        # TODO!!!! To allow initialisation of the hidden state etc. we need to split up
        # the layers into at least input/lstm/output or just keep them all separately!
        # Forget sequence!

        lstm_hiddenunits = 100
        lstm_nlayers = 1
        is_bidirectional = False
        lstmlayer = torch.nn.LSTM(input_size=emblayer.emb_dims,
                                hidden_size=lstm_hiddenunits,
                                num_layers=lstm_nlayers,
                                # dropout=0.5,
                                bidirectional=is_bidirectional,
                                batch_first=True)
        lstmlayer = TakeFromTuple(lstmlayer, which=0)
        lin_units = lstm_hiddenunits*2 if is_bidirectional else lstm_hiddenunits
        self.lstm_totalunits = lin_units
        self.lstm_nlayers = lstm_nlayers
        linlayer = torch.nn.Linear(lin_units, n_classes)
        outlayer = torch.nn.LogSoftmax(dim=1)
        layer_seq = OrderedDict()
        layer_seq['embs'] = emblayer
        layer_seq['lstm'] = lstmlayer
        layer_seq['lin'] = linlayer
        layer_seq['out'] = outlayer
        self.layers = torch.nn.Sequential(layer_seq)
        print("Network: \n", self.layers, file=sys.stderr)
        self.get_init_weights(1)

    # this gets a batch if independent variables
    # By default this is in reshaped padded batch format.
    # For sequences and a single feature this has the format:
    # * a list containing a sublist for each instance
    # * each sublist contains a list with the padded sequence of word indices
    #   (by default the padding index is 0)
    # Note: the calling model wrapper does not automatically put the batch on cuda,
    # so if we want this, it has to be done explicitly in here, using the method
    # on_cuda() to check if cuda is enabled for the module.
    # HOWEVER: we have decided not to put the indices on cuda
    def forward(self, batch):
        # we need only the first feature:
        batch = torch.LongTensor(batch[0])
        if self.on_cuda():
            batch.cuda()
        out, hidden = self.layers(batch)
        return out, hidden

    def get_lossfunction(self, config={}):
        # IMPORTANT: for the target indices, we use -1 for padding by default!
        return torch.nn.NLLLoss(ignore_index=-1)

    def get_optimizer(self, config={}):
        parms = filter(lambda p: p.requires_grad, self.parameters())
        # optimizer = torch.optim.SGD(parms, lr=0.01, momentum=0.9)
        # optimizer = torch.optim.SGD(parms, lr=0.01, momentum=0.9, weight_decay=0.05)
        optimizer = torch.optim.Adam(parms, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        return optimizer

