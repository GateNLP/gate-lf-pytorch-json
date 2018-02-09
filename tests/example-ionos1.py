# simple example of using the ionosphere data directly

from gatelfdata import Dataset
import os
import logging
import torch
import math
from torch.autograd import Variable as V
import torch.nn.functional as F
import torch.nn as nn

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
streamhandler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)
filehandler = logging.FileHandler(__name__+".log")
logger.addHandler(filehandler)

TESTDIR = os.path.join(os.path.dirname(__file__), '.')
DATADIR = os.path.join(TESTDIR, 'data')
TESTFILE1 = os.path.join(DATADIR, "class-ionosphere.meta.json")


ds = Dataset(TESTFILE1)
ds_info = ds.get_info()
logger.info("META: %r" % ds_info)

nFeatures = ds_info["nFeatures"]   # we know they are all numeric!!
nClasses = ds_info["nClasses"] 

hidden = int(math.sqrt(nFeatures))

class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        # first figure out how many inputs we need and also configure the mapper objects for them
        # 1) if there is at least one numeric or binary input we create a linear+nonlin layer for all of them
        #    The mapper object concatenates those features and converts to FloatTensor variables
        #    The output is a bunch of hidden units (same number as inputs?)
        # 2) for each nominal non-sequence input, we check what kind of training is requested:
        #    * if onehot we create a linear layer
        #    * otherwise we check the embeddings id: if we already have seen that id, the existing layer is
        #      reused. Otherwise: if we train the embeddings but do not have a embeddings file, create
        #      a layer and initialize with random vectors. If we do not train and use an embeddings file, create
        #      a constant lookup layer. If we train and use an embeddings file, create our own embeddings mapping layer
        #    Mapper object selects the corresponding feature and converts to LongTensor variables
        #    Output is a hidden layer with either the embeddings or mapped embeddings
        # 3) For a NGRAM, we have a sequence of embedding indices. This creates or re-uses embedding layers as for 2)
        #    For each batch we should have padded sequences, so the shape would be batch,maxseqlen
        #    The embeddings have to specify the padding index!!
        #    Mapper object selects the feature and converts to LongTensor matrix variable
        #    Output is a tensor batchsize,maxseqlen,embsize

        self.lin1 = nn.Linear(nFeatures,hidden)
        self.lin2 = nn.Linear(hidden,nClasses)
        self.final = nn.Softmax()
        self.invar = None

    def forward(self, features_batch):
        # make a single tensor out of all the features
        t1 = torch.FloatTensor(features_batch)
        # we need to transpose the tensor (matrix) since we start off with 
        # the tensor putting our inner lists into rows, but we want to be those in the columns
        # (so we have one row per instance, i.e. the first axis is for the batch)
        t1 = t1.t()
        v1 = V(t1,requires_grad=True)
        self.invar = v1
        tmp1 = self.lin1(v1)
        tmp2 = F.relu(tmp1)
        tmp3 = self.lin2(tmp2)
        out = self.final(tmp3)
        return out

    def get_invar(self):
        return self.invar

model = MyModel()
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# TODO: load validation set and use below for the estimate

for epoch in range(10):
    for b in ds.batches_converted(convertedFile=ds.converted4meta(TESTFILE1), as_numpy=True, batch_size=5):
        # logger.info("BATCH: %r" % (b,))
        (indep,dep) = b
        pred=model(indep)
        # make a variable out of the target
        tt = torch.FloatTensor(dep)
        target = V(tt)
        loss = loss_fn(pred,target)
        logger.info("LOSS: %s" % loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        v1 = model.get_invar()
        # logger.info("GRAD: %s" % v1.grad)

