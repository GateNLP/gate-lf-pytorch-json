import torch
from torch.autograd import Variable as V
import sys
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)


class EmbeddingsModule(torch.nn.Module):

    def __init__(self, vocab, cuda=None, emb_dims=None):
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
        self.emb_file = vocab.emb_file
        self.modulename = "embeddings:{}:{}:{}:{}:{}".format(
            self.emb_id, self.emb_dims, self.emb_train, self.emb_minfreq, self.emb_file)
        self.modulename = self.modulename.replace(".", "_")
        weights = torch.from_numpy(vocab.get_embeddings())
        # TODO: check if there is any difference/advantage of using Embedding.from_pretrained(embs, freeze=True/False)
        # (this does not allow to set the padding idx!!)
        module = torch.nn.Embedding(self.emb_size, embedding_dim=self.emb_dims, padding_idx=0, _weight=weights)
        if self.emb_train == "no" or self.emb_train == "mapping":
            # TODO: this was False, changed to True, because False showed much worse performance!
            module.weight.requires_grad = True
        # if we have a mapping, we learn a nonlinear mapping from the constant embedding vector to our internal
        # representation which has the exact same number of dimensions
        # IMPORTANT: the mapping weights also need to get shared between all mapping layers!
        # Currently we achieve this by storing the parameters in the vocab instance as a transient
        # field
        if self.emb_train == "mapping":
            mappinglayer = torch.nn.Linear(self.emb_dims, self.emb_dims)
            if hasattr(vocab, "_mappingparms"):
                mappinglayer.weight = vocab._mappingparms
            else:
                vocab._mappingparms = mappinglayer.weight
            module = torch.nn.Sequential(module, mappinglayer, torch.nn.Sigmoid())
        self.add_module(self.modulename, module)
        self.modules = [module]
        self._on_cuda = cuda

    def on_cuda(self):
        """Returns true or false depending on if the module is on cuda or not. Unfortunately
        there is no API method in PyTorch for this so we get this from the first parameter of the
        model and cache it."""
        if self._on_cuda is None:
            self._on_cuda = next(self.parameters()).is_cuda
            # if we actually are on cuda, make sure all the modules are on cuda as well!
            if self._on_cuda:
                for module in self.modules:
                    module.cuda()
        return self._on_cuda

    def forward(self, batch):
        # TODO: eventually, we should decide if an embeddings layer should be on the GPU or not.
        # Then we also need to place the tensor on the GPU or not before runninng the embeddings layer.
        # This should depend on several factos, embeddings matrix size, if we learn etc.
        # To make it work properly, we also need to make sure that the top-level .cuda() invocation does
        # not enable cuda for the module in here unless we really want it so we need to override the method.
        # NOTE: for now we run on the cuda, if enabled

        # NOTE: we already get a tensor here
        # TODO: not sure if we can get a float tensor here, if yes, we need to convert to a long tensor
        # print("DEBUG: type of batch=", type(batch), file=sys.stderr)
        batch_var = V(torch.LongTensor(batch), requires_grad=False)
        if self.on_cuda():
            batch_var = batch_var.cuda()
        out = self.modules[0](batch_var)
        return out

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_on_cuda"]
        return state

    def __setstate__(self, state):
        state["_on_cuda"] = None
        self.__dict__.update(state)
