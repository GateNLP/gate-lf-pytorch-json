import torch.nn
from gatelfpytorchjson import CustomModule
from gatelfpytorchjson import EmbeddingsModule
import sys


class SeqTaggerLSTMSimple(CustomModule):

    def get_init_weights(self, batchsize):
        """A method that returns a tensor that can be used to initialize the hidden states of the LSTM.
        """
        # first get the parameters so we can create the same type of vector, from the same device easily
        # the lstm is really wrapped in a a "takefromtuple" layer, so we have to get it out of there
        lstmparms = next(self.layer_lstm.parameters()).data
        h0 = lstmparms.new(self.layer_lstm.num_layers, batchsize, self.layer_lstm.hidden_size)
        c0 = lstmparms.new(self.layer_lstm.num_layers, batchsize, self.layer_lstm.hidden_size)
        h0.copy_(torch.randn(self.layer_lstm.num_layers, batchsize, self.layer_lstm.hidden_size)*0.01)
        c0.copy_(torch.randn(self.layer_lstm.num_layers, batchsize, self.layer_lstm.hidden_size)*0.01)
        return h0, c0

    # NOTE: make sure the dataset is not stored and only used to decide on parameters etc so
    # that the dataset data is not getting pickled when the model is saved!
    def __init__(self, dataset, config={}):
        super().__init__(config=config)

        self.n_classes = dataset.get_info()["nClasses"]
        # create a simple LSTM-based network: the input uses an embedding layer created from
        # the vocabulary, then a bidirectional LSTM followed by a simple softmax layer for each element
        # in the sequence
        feature = dataset.get_index_features()[0]
        vocab = feature.vocab
        # create the embedding layer from the vocab
        self.layer_emb = EmbeddingsModule(vocab)

        # for debugging, save the vocab here
        self.vocab = vocab

        self.lstm_hiddenunits = 100
        self.lstm_nlayers = 1
        self.lstm_is_bidirectional = False
        self.layer_lstm = torch.nn.LSTM(input_size=self.layer_emb.emb_dims,
                                        hidden_size=self.lstm_hiddenunits,
                                        num_layers=self.lstm_nlayers,
                                        dropout=0.5,
                                        bidirectional=self.lstm_is_bidirectional,
                                        batch_first=True)
        lin_units = self.lstm_hiddenunits*2 if self.lstm_is_bidirectional else self.lstm_hiddenunits
        self.lstm_totalunits = lin_units
        self.layer_lin = torch.nn.Linear(lin_units, self.n_classes)
        self.layer_out = torch.nn.LogSoftmax(dim=1)
        print("Network: \n", self, file=sys.stderr)

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
        # print("DEBUG: batch=", batch, file=sys.stderr)
        batch = torch.LongTensor(batch[0])
        print("DEBUG: batch size=", batch.size(), file=sys.stderr)
        # debug: try to get the words for each of the indices in the batch
        # this is of shape batchsize,maxseqlength
        sequences = []
        for i in range(batch.size()[0]):
            sequence = []
            for j in range(batch.size()[1]):
                sequence.append(self.vocab.itos[batch[i,j]])
            sequences.append(sequence)

        print("DEBUG: sequences=", sequences, file=sys.stderr)
        if self.on_cuda():
            batch.cuda()
        tmp = self.layer_emb(batch)
        # print("DEBUG: embout=", tmp, file=sys.stderr)
        hidden_init = self.get_init_weights(len(batch))
        # print("DEBUG: lstm input size=", tmp.size(), file=sys.stderr)
        # print("DEBUG: hidden_init[0] size=", hidden_init[0].size(), file=sys.stderr)
        # print("DEBUG: hidden_init[1] size=", hidden_init[1].size(), file=sys.stderr)
        tmp, hidden = self.layer_lstm(tmp, hidden_init)
        # tmp, hidden = self.layer_lstm(tmp)

        tmp = self.layer_lin(tmp)
        out = self.layer_out(tmp)
        sys.exit()
        return out

    def get_lossfunction(self, config={}):
        # IMPORTANT: for the target indices, we use -1 for padding by default!
        return torch.nn.NLLLoss(ignore_index=-1)

    def get_optimizer(self, config={}):
        parms = filter(lambda p: p.requires_grad, self.parameters())
        # optimizer = torch.optim.SGD(parms, lr=0.01, momentum=0.9)
        # optimizer = torch.optim.SGD(parms, lr=0.01, momentum=0.9, weight_decay=0.05)
        optimizer = torch.optim.Adam(parms, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        return optimizer

