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
        self.num_feats = dataset.get_numeric_features()
        self.nom_feats = dataset.get_nominal_features()
        self.ngr_feats = dataset.get_ngram_features()
        self.featureinfo = {"num_idxs": self.num_idxs,
                            "nom_idxs": self.nom_idxs,
                            "ngr_idxs": self.ngr_idxs}
        self.info = dataset.get_info()
        if self.info["isSequence"]:
            raise Exception("Sequence tagging not yet implemented")
        else:
            if self.info["targetType"] == "nominal":
                self.init_classification(dataset)
            else:
                raise Exception("Target type not yet implemented: %s" % self.info["targetType"])

    def init_classification(self, dataset):
        n_classes = self.info["nClasses"]
        inputlayers = []
        # keep track of the number of input layer output dimensions
        inlayers_outdims = 0
        # if we have numeric features, create the numeric input layer
        if len(self.num_idxs) > 0:
            n_in = len(self.num_idxs)
            n_hidden = ModelWrapper.makeless(n_in)
            lin = torch.nn.Linear(n_in, n_hidden)
            drp = torch.nn.Dropout(p=0.2)
            act = torch.nn.ELU()
            layer = torch.nn.Sequential(lin,drp,act)
            inlayers_outdims += n_hidden
            lname = "input_numeric"
            inputlayers.append((layer, {"type": "numeric", "name": lname}))
            pass
        # if we have nominal features, create all the layers for those
        # TODO: may need to handle onehot features differently!!
        # remember which layers we already have for an embedding id
        nom_layers = {}
        for i in range(len(self.nom_feats)):
            nom_feat = self.nom_feats[i]
            nom_idx = self.nom_idxs[i]
            # depending on what kind of training is defined and
            # what the embedding id / vocabulary for that feature is,
            # we create or re-use one of several possible kinds of layers

            raise Exception("Support for nomunal inputs not yet implemented")
        for i in range(len(self.ngr_feats)):
            ngr_feat = self.ngr_feats[i]
            nom_idx = self.ngr_idxs[i]
            # depending on the training defined we use an LSTM subnetwork
            # based on one of the nominal layers we have.
            raise Exception("Support for ngram inputs not yet implemented")

        # Now create the hidden layers
        hiddenlayers = []
        # for now, one hidden layer for compression and another
        # to map to the number of classes
        n_hidden1lin_out = ModelWrapper.makeless(inlayers_outdims)
        hidden1lin = torch.nn.Linear(inlayers_outdims, n_hidden1lin_out)
        hidden1drp = torch.nn.Dropout(p=0.2)
        hidden1act = torch.nn.ELU()
        hidden2 = torch.nn.Linear(n_hidden1lin_out, n_classes)
        hidden = torch.nn.Sequential(hidden1lin, hidden1drp,
                                     hidden1act, hidden2)
        hiddenlayers.append((hidden, {"name": "hidden"}))
        # Create the output layer
        out = torch.nn.Softmax()
        outputlayer = (out, {"name": "output"})
        # create the module and store it
        self.module = ClassificationModelSimple(inputlayers,
                                                hiddenlayers,
                                                outputlayer,
                                                self.featureinfo)
        # Decide on the loss function here for training later!
        self.loss = torch.nn.CrossEntropyLoss()


    def get_module(self):
        return self.module

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