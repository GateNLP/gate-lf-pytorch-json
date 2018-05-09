import torch
from torch.autograd import Variable as V
from collections import defaultdict

class EmbeddingsModule(torch.nn.Module):

    def __init__(self, vocab, cuda=None):
        """If vocab.train is yes new embeddings will get learned, starting off with random vectors if no pretrained
        embeddings are given, otherwise the pretrained embeddings will be used where possible.
        If train is no, then no training will be done and the pretrained embeddings will be used only.
        If train is mapping then a mapping is learned from pretrained embeddings to our own embeddings.
        NOTE: this should all happen automatically by inspecting and using the vocab instance.
        NOTE: the actual embeddings getting loaded into the Embeddings module are derived from the vocab,
        but if there is already an Embedding module for the same emb_id, the weights are shared.
        """

        super().__init__()
        self.emb_id = vocab.emb_id
        self.emb_train = vocab.emb_train
        self.emb_dims = vocab.emb_dims
        self.emb_minfreq = vocab.emb_minfreq
        self.emb_size = vocab.n
        self.modulename = "embeddings:{}:{}:{}:{}".format(self.emb_id, self.emb_dims, self.emb_train, self.emb_minfreq)
        weights = torch.from_numpy(vocab.get_embeddings())
        module = torch.nn.Embedding(self.emb_size, embedding_dim=self.emb_dims, padding_idx=0, _weight=weights)
        if self.emb_train == "no" or self.emb_train == "mapping":
            module.weight.requires_grad = False
        # if we have a mapping, we learn a nonlinear mapping from the constant embedding vector to our internal
        # representation which has the exact same number of dimensions
        if self.emb_train == "mapping":
            module = torch.nn.Sequential(module, torch.nn.Linear(self.emb_size, self.emb_size), torch.nn.Sigmoid())
        self.add_module(self.modulename, module)
        self.modules = [module]
        self._on_cuda = cuda

    def on_cuda(self):
        """Returns true or false depending on if the module is on cuda or not. Unfortunately
        there is no API method in PyTorch for this so we get this from the first parameter of the
        model and cache it."""
        if self._on_cuda is None:
            self._on_cuda = next(self.parameters()).is_cuda
        return self._on_cuda

    def forward(self, batch):
        # TODO: eventually, we should decide if an embeddings layer should be on the GPU or not.
        # Then we also need to place the tensor on the GPU or not before runninng the embeddings layer.
        # This should depend on several factos, embeddings matrix size, if we learn etc.
        # To make it work properly, we also need to make sure that the top-level .cuda() invocation does
        # not enable cuda for the module in here unless we really want it so we need to override the method.
        # NOTE: for now we run on the cuda, if enabled
        batch_var = V(torch.LongTensor(batch), requires_grad=False)
        if self.on_cuda():
            batch_var = batch_var.cuda()
        out = self.modules[0](batch_var)
        return out
