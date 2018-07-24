import torch.nn
from gatelfpytorchjson import CustomModule
from gatelfpytorchjson import EmbeddingsModule
from gatelfpytorchjson import TakeFromTuple
from torch.autograd import Variable as V
import torch.nn.functional as F
import sys


class SeqTaggerLSTMSimple(CustomModule):

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

        lstmlayer = torch.nn.LSTM(input_size=emblayer.emb_dims,
                                hidden_size=200,
                                num_layers=1,
                                # dropout=0.1,
                                bidirectional=True,
                                batch_first=True)
        lstmlayer = TakeFromTuple(lstmlayer, which=0)
        linlayer = torch.nn.Linear(200*2, n_classes)
        out = torch.nn.LogSoftmax(dim=1)
        self.layers = torch.nn.Sequential(
            emblayer,
            lstmlayer,
            linlayer,
            out
        )
        print("Network: \n", self.layers, file=sys.stderr)

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
        out = self.layers(batch)
        return out
