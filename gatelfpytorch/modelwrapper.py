import torch
import torch.nn
import math
from torch.autograd import Variable as V
import torch.nn.functional as F
from .classificationmodelsimple import ClassificationModelSimple

# Basic usage:
# ds = Dataset(metafile)
# wrapper = ModelWrapperSimple(ds) # or some other subclass
# wrapper.train()
# # get some data for application some where
# instances = get_them()
# preditions = wrapper.apply(instances)
# NOTE: maybe use the same naming conventions as scikit learn here!!

class ModelWrapper(object):

    @staticmethod
    def makeless(n, func=math.pow, preshift=-1.0, postshift=1.0, p1=0.5):
        val = int(func((n+preshift),p1)+postshift)
        return val

    # This requires an initialized dataset instance
    def __init__(self, dataset):
        self.num_idxs = dataset.get_numeric_feature_idxs()
        self.nom_idxs = dataset.get_nominal_feature_idxs()
        self.ngr_idxs = dataset.get_ngram_feature_idxs()
        num_feats = dataset.get_numeric_features()
        nom_feats = dataset.get_nominal_features()
        ngr_feats = dataset.get_ngram_features()
        self.featureinfo = {"num_idxs": self.num_idxs,
                            "nom_idxs": self.nom_idxs,
                            "ngr_idxs": self.ngr_idxs}
        inputlayers = []
        # if we have numeric features, create the numeric input layer
        if len(self.num_idxs) > 0:
            n_in = len(self.num_idxs)
            n_hidden = ModelWrapper.makeless(n_in)
            lin = torch.nn.Linear(n_in, n_hidden)
            drp = torch.nn.Dropout(p=0.2)
            act = torch.nn.ELU()
            layer = torch.nn.Sequential(lin,drp,act)
            lname = "input_numeric"
            inputlayers.append((layer, {"type": "numeric", "name": lname}))
            pass
        # if we have nominal features, create all the layers for those
        # TODO: may need to handle onehot features differently!!
        for i in range(len(nom_feats)):
            # nom_feat = nom_feats[i]
            # nom_idx = self.nom_idxs[i]
            # TODO!!!
            raise Exception("Support for nomunal inputs not yet implemented")
        for i in range(len(ngr_feats)):
            raise Exception("Support for ngram inputs not yet implemented")
        # Now create the hidden layers
        hiddenlayers = []
        # Create the output layer
        outputlayer = None
        # create the module and store it
        self.module = ClassificationModelSimple(inputlayers,
                                                hiddenlayers,
                                                outputlayer,
                                                self.featureinfo)
        # Decide on the loss function here for training later!


    # the implementation should figure out best values if parameter
    # is set to None
    # Also, by default, the method should decide which format
    # to use for reading the data (original or converted)
    def train(self, epochs=None, batchsize=None):
        # make sure we are in training mode

        # get the validation set
        # iterate over epochs
        #   iterate over batches
        #     train on the batch
        #     get train evaluation and log
        #     every K, get validation evaluation and log
        #     if early stopping criterion, stop if satisfied
        raise Exception("Not yet implemented")