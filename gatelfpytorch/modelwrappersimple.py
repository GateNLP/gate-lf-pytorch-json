from . modelwrapper import ModelWrapper
from . embeddingsmodule import EmbeddingsModule
import torch
import torch.nn
import torch.optim
import math
from torch.autograd import Variable as V
import torch.nn.functional as F
from .classificationmodelsimple import ClassificationModelSimple
import logging
import sys

# Basic usage:
# ds = Dataset(metafile)
# wrapper = ModelWrapperSimple(ds) # or some other subclass
# wrapper.train()
# # get some data for application some where
# instances = get_them()
# preditions = wrapper.apply(instances)
# NOTE: maybe use the same naming conventions as scikit learn here!!


class ModelWrapperSimple(ModelWrapper):


    # This requires an initialized dataset instance
    def __init__(self, dataset, config=None):
        """This requires a gatelfdata Dataset instance and can optionally take a dictionary with
        configuration/initialization options (NOT SUPPORTED YET)"""
        super().__init__(dataset)
        self.dataset = dataset
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
        self.module = None # the init_<TASK> method actually sets this
        if self.info["isSequence"]:
            raise Exception("Sequence tagging not yet implemented")
        else:
            if self.info["targetType"] == "nominal":
                self.init_classification(dataset)
            else:
                raise Exception("Target type not yet implemented: %s" % self.info["targetType"])
        self.optimizer = torch.optim.SGD(self.module.parameters(), lr=0.001, momentum=0.9)
        # various configuration settings
        self.validate_every_batches = 100
        self.validate_every_epochs = None

    def init_classification(self, dataset):
        n_classes = self.info["nClasses"]
        inputlayers = []
        # keep track of the number of input layer output dimensions
        inlayers_outdims = 0
        # if we have numeric features, create the numeric input layer
        if len(self.num_idxs) > 0:
            n_in = len(self.num_idxs)
            n_hidden = ModelWrapper.makeless(n_in, p1=0.5)
            lin = torch.nn.Linear(n_in, n_hidden)
            drp = torch.nn.Dropout(p=0.2)
            act = torch.nn.ELU()
            layer = torch.nn.Sequential(lin, drp, act)
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
            vocab = nom_feat.vocab
            emb_id = vocab.emb_id
            if emb_id in nom_layers:
                emblayer = nom_layers.get(emb_id)
            else:
                emblayer = EmbeddingsModule(vocab)
                nom_layers[emb_id] = emblayer
            lname = "input_emb_%s_%s" % (i, emb_id)
            inputlayers.append((emblayer, {"type": "nominal", "name": lname}))
            inlayers_outdims += emblayer.emb_dim
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
        out = torch.nn.Softmax(dim=1)
        outputlayer = (out, {"name": "output"})
        # create the module and store it
        self.module = ClassificationModelSimple(inputlayers,
                                                hiddenlayers,
                                                outputlayer,
                                                self.featureinfo)
        # Decide on the loss function here for training later!
        self.loss = torch.nn.CrossEntropyLoss()


    def get_module(self):
        """Return the PyTorch module that has been built and is used by this wrapper."""
        return self.module

    # the implementation should figure out best values if parameter
    # is set to None
    # Also, by default, the method should decide which format
    # to use for reading the data (original or converted)
    def train(self, max_epochs=None, batch_size=None, validationsize=None, early_stopping=True):
        """Train the model on the dataset. max_epochs is the maximum number of
        epochs to train, but if early_stopping is enabled, it could be fewer.
        If early_stopping is True, a default early stopping strategy is used,
        if set to a function that function (taking a last of recent evaluations
        and returning boolean) is used. The batchsize parameter can be used
        to override the batchsize, similar the validationsize parameter to
        override the validation set size (if float, the portion, if int the
        numer of instances)"""
        if not max_epochs:
            # TODO: need some clever way to set the epochs here
            max_epochs = 10000
        if early_stopping:
            if isinstance(early_stopping, bool):
                early_stopping_function = ModelWrapper.early_stopping_checker
            else:
                early_stopping_function = early_stopping
        logger = logging.getLogger(__name__)
        # make sure we are in training mode
        self.module.train(mode=True)
        if not batch_size:
            batch_size = 10
        # get the validation set
        valsize = None
        valpart = 0.1
        if validationsize:
            if isinstance(validationsize, int):
                valsize = validationsize
            elif isinstance(validationsize, float):
                valpart = validationsize
            else:
                raise Exception("Parameter validationsize must be a float or int")
        self.dataset.split(convert=True, validation_part=valpart, validation_size=valsize)
        valset = self.dataset.validation_set_converted(as_batch=True)
        val_indeps = valset[0]
        val_targets = V(torch.LongTensor(valset[1]), requires_grad=False)
        stop_it_already = False
        validation_losses = []
        print("DEBUG: before running training...", file=sys.stderr)
        for epoch in range(1, max_epochs+1):
            batch_nr = 0
            for batch in self.dataset.batches_converted(train=True, batch_size=batch_size):
                batch_nr += 1
                self.module.zero_grad()
                # train on a whole batch
                # step 1: run the data through
                output = self.module(batch[0])
                # step 2: calculate the loss
                targets = V(torch.LongTensor(batch[1]), requires_grad=False)
                # print("DEBUG: targets = ", list(targets), "out=", list(output), file=sys.stderr)
                loss = self.loss(output, targets)
                # calculate the accuracy as well
                acc = ModelWrapper.accuracy(output, targets)
                logger.debug("Batch loss/acc for epoch=%s, batch=%s: %s / %s" % (epoch, batch_nr, float(loss), acc))
                # print("Batch loss/acc for epoch=%s, batch=%s: %s / %s" % (epoch, batch_nr, float(loss), acc), file=sys.stderr)
                loss.backward()
                self.optimizer.step()
                if (self.validate_every_batches and ((batch_nr % self.validate_every_batches) == 0)) or\
                        (self.validate_every_epochs and ((epoch % self.validate_every_epochs) == 0)):
                    batch_nr += 1
                    # evaluate on validation set
                    self.module.eval()
                    out_val = self.module(val_indeps)
                    loss_val = self.loss(out_val, val_targets)
                    acc_val = ModelWrapper.accuracy(out_val, val_targets)
                    validation_losses.append(float(loss_val))
                    # if we have early stopping, check if we should stop
                    var = None
                    if early_stopping:
                        (stop_it_already, var) = early_stopping_function(validation_losses)
                        if stop_it_already:
                            logger.info("Early stopping criterion reached, stopping training, var=%s" % var)
                    logger.info("Evaluation epoch=%s,batch=%s,train-loss/val-loss/val-variance/train-acc/val-acc: %s / %s / %s / %s / %s" %
                                (epoch, batch_nr, float(loss), float(loss_val), var, acc, acc_val))
                if stop_it_already:
                    break
            if stop_it_already:
                break
