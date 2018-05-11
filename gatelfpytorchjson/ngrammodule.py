from . embeddingsmodule import EmbeddingsModule
import torch

class NgramModule(torch.nn.Module):

    def __init__(self, embeddingsmodule, method="lstm"):
        """Create a module for processing ngram sequences for the given EmbeddingsModule embeddingsmodule.
        How eventually will be one of  lstm, gru, conv, mean, sum.
        TODO: currently only lstm is supported!"""
        super().__init__()
        # since we directly assign, this should get registered!
        self.embeddingsmodule = embeddingsmodule
        # TODO: this will later select and configure the method
        self.forward_method = self.forward_method_lstm
        self.init_method = self.init_method_lstm
        # now use the configured init method
        self.init_method()

    def init_method_lstm(self):
        # TODO: maybe use heuristics to find better values for
        # hidden_size
        # num_layers
        # dropout
        # bidirectional
        num_layers = 1
        bidirectional = True
        hidden_size = self.embeddingsmodule.emb_dims
        self.lstm = torch.nn.LSTM(input_size=self.embeddingsmodule.emb_dims,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  # dropout=0.1,
                                  bidirectional=bidirectional,
                                  batch_first=True)
        # TODO: create a better lstm initialisation vector here for repeated
        # use doring forward, if needed!
        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1
        self.out_dim = hidden_size * num_layers * num_directions

    def forward_method_lstm(self, batchofsequences):
        batchsize = len(batchofsequences)
        # NOTE: we already expect batchofsequences to be a variable with batch_first zero-padded sequences!
        # now run the data through the embeddings, then run the sequences of embeddings through the lstm
        # Note: the embeddingsmodule.forward method expects the original batchofsequences, we do not need to convert
        # to a tensor and variable here!
        tmp1 = self.embeddingsmodule.forward(batchofsequences)
        out, (h0, c0) = self.lstm(tmp1)  # TODO: for now we use zero vectors for initialization
        # we only need the final hidden state
        return h0.view(batchsize, -1)

    def forward(self, batchofsequences):
        # just delegate to the forward method for the method chosen
        return self.forward_method(batchofsequences)