import torch
from torch.autograd import Variable as V


class EmbeddingsModule(torch.nn.Module):

    def __init__(self, vocab):
        """If vocab.train is yes new embeddings will get learned, starting off with random vectors if no pretrained
        embeddings are given, otherwise the pretrained embeddings will be used where possible.
        If train is no, then no training will be done and the pretrained embeddings will be used only.
        If train is mapping then a mapping is learned from pretrained embeddings to our own embeddings.
        NOTE: this should all happen automatically by inspecting and using the vocab instance.
        """
        # TODO: if we have embeddings, load them
        # TODO: if we do not have embeddings, update vocab OOV from Embeddings or vice versa
        # TODO: mapping vocab support not fully implemented yet (need to add embeddings words)

        super().__init__()
        self.emb_id = vocab.emb_id
        self.emb_train = vocab.emb_train
        self.emb_dims = vocab.emb_dims
        if not self.emb_dims:
            self.emb_dims = 100
        self.emb_size = vocab.n
        self.module = torch.nn.Embedding(self.emb_size, embedding_dim=self.emb_dims, padding_idx=0)
        self._on_cuda = None

    def on_cuda(self):
        """Returns true or false depending on if the module is on cuda or not. Unfortunately
        there is no API method in PyTorch for this so we get this from the first parameter of the
        model and cache it."""
        if self._on_cuda is None:
            self._on_cuda = next(self.parameters())
        return self._on_cuda

    def forward(self, batch):
        # for space reasons, for now we run the embedding layer on the cpu, then pass the rest on to the GPU
        # if cuda should be enbaled
        batch_var = V(torch.LongTensor(batch), requires_grad=False)
        out = self.module(batch_var)
        if self.on_cuda():
            out = out.cuda()
        return out
