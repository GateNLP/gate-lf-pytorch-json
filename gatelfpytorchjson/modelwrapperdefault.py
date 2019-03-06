from gatelfpytorchjson.modelwrapper import ModelWrapper
from gatelfpytorchjson.embeddingsmodule import EmbeddingsModule
from gatelfpytorchjson.ngrammodule import NgramModule
import os
import torch
import torch.nn
import torch.optim
from .classificationmodule import ClassificationModule
from .takefromtuple import TakeFromTuple
import sys
import pickle
from gatelfdata import Dataset
import numpy as np
import pkgutil
import timeit
import logging
import signal


# Basic usage:
# ds = Dataset(metafile)
# wrapper = ModelWrapperSimple(ds) # or some other subclass
# wrapper.train()
# # get some data for application some where
# instances = get_them()
# preditions = wrapper.apply(instances)
# NOTE: maybe use the same naming conventions as scikit learn here!!

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(levelname)-1s %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)


def f(value):
    """Format a float value to have 3 digits after the decimal point"""
    return "{0:.3f}".format(value)


class ModelWrapperDefault(ModelWrapper):

    def init_from_dataset(self):
        """Set the convenience attributes which we get from the dataset instance"""
        dataset = self.dataset
        self.metafile = dataset.metafile
        self.float_idxs = dataset.get_float_feature_idxs()
        self.index_idxs = dataset.get_index_feature_idxs()
        self.indexlist_idxs = dataset.get_indexlist_feature_idxs()
        self.float_feats = dataset.get_float_features()
        self.index_feats = dataset.get_index_features()
        self.indexlist_feats = dataset.get_indexlist_features()
        self.featureinfo = {"num_idxs": self.float_idxs,
                            "nom_idxs": self.index_idxs,
                            "ngr_idxs": self.indexlist_idxs}
        self.info = dataset.get_info()

    # This requires an initialized dataset instance
    def __init__(self, dataset, config={}, cuda=None):
        """This requires a gatelfdata Dataset instance and can optionally take a dictionary with
        configuration/initialization options (NOT SUPPORTED YET).
        If cuda is None, then if cuda is available it will be used. True and False
        require and prohibit the use of cuda unconditionally.
        Config settings: stopfile: a file path, if found training is stopped
        """
        super().__init__(dataset, config=config)
        self.config = config
        logger.debug("Init with config=%s" % (config,))
        if "cuda" in config and config["cuda"] is not None:
            cuda = config["cuda"]
        self.cuda = cuda
        self.checkpointnr = 0
        self.stopfile = os.path.join(os.path.dirname(dataset.metafile), "STOP")
        if "stopfile" in config and config["stopfile"] is not None:
            self.stopfile = config["stopfile"]
        self.stopfile = os.path.abspath(self.stopfile)
        # logging.getLogger(__name__).debug("Set the stop file to %s" % self.stopfile)
        self.override_learningrate = None
        if "learningrate" in config and config["learningrate"]:
            self.override_learningrate = config["learningrate"]
        cuda_is_available = torch.cuda.is_available()
        if self.cuda is None:
            enable_cuda = cuda_is_available
        else:
            enable_cuda = self.cuda
        self._enable_cuda = enable_cuda  # this tells us if we should actually set cuda or not
        logger.debug("Init cuda=%s enable_cuda=%s" % (cuda, self._enable_cuda,))
        self.dataset = dataset
        self.init_from_dataset()
        # various configuration settings which can be set before passing on control to the
        # task-speicific initialization
        self.best_model_saved = False
        self.validate_every_batches = None
        self.validate_every_epochs = 1
        self.validate_every_instances = None
        self.report_every_batches = None
        self.report_every_instances = 500
        self.is_data_prepared = False
        self.valset = None   # Validation set created by prepare_data
        self.lossfunction = None
        self.module = None  # the init_<TASK> method actually sets this!!
        self.random_seed = config.get("seed", 0)
        torch.manual_seed(self.random_seed)
        # make sure it is set on all GPUs as well, we can always do this as torch ignores
        # this if no CUDA is available
        torch.cuda.manual_seed_all(self.random_seed)
        # if the config requires a specific module needs to get used, create it here, otherwise
        # create the module needed for sequences or non-sequences
        # IMPORTANT! the optimizer needs to get created after the module has been moved to a GPU
        # using cuda()!!!
        if "module" in config and config["module"] is not None:
            # logger.debug("Init, modules importable: %s" %
            #              ([x[1] for x in pkgutil.iter_modules(path=".gatelfpytorchjson")],))
            # TODO: figure out how to do this right!!
            ptclassname = config["module"]
            logger.debug("Init import, trying to use class/file: %s" % (ptclassname,))
            import importlib

            # NOTE:
            # the following worked and seemed to be required on one computer ...
            # parent = importlib.import_module(".."+ptclassname, package=".gatelfpytorchjson.modules."+ptclassname)
            # this works fine:
            parent = importlib.import_module("gatelfpytorchjson.modules."+ptclassname)

            class_ = getattr(parent, ptclassname)
            self.module = class_(dataset, config=config)
            # TODO: best method to configure the loss for the module? for now we expect a static method
            # in the class that returns it
            self.lossfunction = self.module.get_lossfunction(config=config)
            if self._enable_cuda:
                self.module.cuda()
                self.lossfunction.cuda()
            self.optimizer = self.module.get_optimizer(config=config)
        else:
            if self.info["isSequence"]:
                self.init_sequencetagging(dataset)
            else:
                if self.info["targetType"] == "nominal":
                    self.init_classification(dataset)
                else:
                    raise Exception("Target type not yet implemented: %s" % self.info["targetType"])
            if self._enable_cuda:
                self.module.cuda()
                self.lossfunction.cuda()
            # get the parameters for the optimizer, but make sure we do not include parameters for fixed layers!
            params = filter(lambda p: p.requires_grad, self.module.parameters())
            # self.optimizer = torch.optim.SGD(self.module.parameters(), lr=0.001, momentum=0.9)
            # self.optimizer = torch.optim.SGD(self.module.parameters(), lr=(self.override_learningrate or 0.001))
            # self.optimizer = torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
            # self.optimizer = torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
            self.optimizer = torch.optim.Adam(params, lr=(self.override_learningrate or 0.001), betas=(0.9, 0.999), eps=1e-08, weight_decay=0 )
            # self.optimizer = torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
            # self.optimizer = torch.optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
            # self.optimizer = torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
            # self.optimizer = torch.optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
            # self.optimizer = torch.optim.SGD(params, lr=0.1, momentum=0, dampening=0, weight_decay=0, nesterov=False)
            # NOTE/TODO: check out how to implement a learning rate scheduler that makes the LR depend e.g. on epoch, see
            # http://pytorch.org/docs/master/optim.html
            # e.g. every 10 epochs, make lr half of what it was:
            # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
            # self.optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.0)
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)

    def get_logger(self):
        """
        Expose whatever logger instance this class is using
        """
        return logger

    # This is mainly used at application time, for training, the same thing happens in init.
    # TODO: this should get moved into a common superclass for all modelwrappers!
    def set_cuda(self, flag):
        """Advise to use CUDA if flag is True, or CPU if false. True is ignored if cuda is not available"""
        if flag and torch.cuda.is_available():
            self.module.cuda()
            self.lossfunction.cuda()
            self._enable_cuda = True
        else:
            self.module.cpu()
            self.lossfunction.cpu()
            self._enable_cuda = False


    def _signal_handler(self, sig, frame):
        logger.info("Received interrupt signal, setting interrupt flag")
        self.interrupted = True

    def init_classification(self, dataset):
        n_classes = self.info["nClasses"]
        inputlayers = []
        # keep track of the number of input layer output dimensions
        inlayers_outdims = 0
        # if we have numeric features, create the numeric input layer
        if len(self.float_idxs) > 0:
            n_in = len(self.float_idxs)
            # n_hidden = ModelWrapper.makeless(n_in, p1=0.5)
            n_hidden = n_in
            lin = torch.nn.Linear(n_in, n_hidden)
            act = torch.nn.ELU()
            # act = torch.nn.Softmax(dim=1)
            layer = torch.nn.Sequential(lin, act)
            inlayers_outdims += n_hidden
            lname = "input_numeric"
            inputlayers.append((layer, {"type": "numeric", "name": lname}))
            pass
        # if we have nominal features, create all the layers for those
        # TODO: may need to handle onehot features differently!!
        # remember which layers we already have for an embedding id
        nom_layers = {}
        for i in range(len(self.index_feats)):
            nom_feat = self.index_feats[i]
            nom_idx = self.index_idxs[i]
            vocab = nom_feat.vocab
            emb_id = vocab.emb_id
            if emb_id in nom_layers:
                emblayer = nom_layers.get(emb_id)
            else:
                emblayer = EmbeddingsModule(vocab)
                nom_layers[emb_id] = emblayer
            lname = "input_emb_%s_%s" % (i, emb_id)
            inputlayers.append((emblayer, {"type": "nominal", "name": lname}))
            inlayers_outdims += emblayer.emb_dims
        for i in range(len(self.indexlist_feats)):
            ngr_feat = self.indexlist_feats[i]
            nom_idx = self.indexlist_idxs[i]
            vocab = ngr_feat.vocab
            emb_id = vocab.emb_id
            if emb_id in nom_layers:
                emblayer = nom_layers.get(emb_id)
            else:
                emblayer = EmbeddingsModule(vocab)
                nom_layers[emb_id] = emblayer
            lname = "input_ngram_%s_%s" % (i, emb_id)
            ngramlayer = NgramModule(emblayer, config=self.config)  # setting ngram_layer: cnn or lstm
            inputlayers.append((ngramlayer, {"type": "ngram", "name": lname}))
            inlayers_outdims += ngramlayer.out_dim
        # Now create the hidden layers
        hiddenlayers = []

        # THIS WAS THE OLD APPROACH, using TWO linear layers, separated by ELU
        # for now, one hidden layer for compression and another
        # to map to the number of classes
        #n_hidden1lin_out = ModelWrapper.makeless(inlayers_outdims)
        #hidden1lin = torch.nn.Linear(inlayers_outdims, n_hidden1lin_out)
        #hidden1act = torch.nn.ELU()
        #hidden2 = torch.nn.Linear(n_hidden1lin_out, n_classes)
        #hidden = torch.nn.Sequential(hidden1lin,
        #                             hidden1act, hidden2)

        # INSTEAD we just use a single linear layer, no nonlinearity
        hidden = torch.nn.Linear(inlayers_outdims, n_classes)

        hiddenlayers.append((hidden, {"name": "hidden"}))
        # Create the output layer
        out = torch.nn.LogSoftmax(dim=1)
        outputlayer = (out, {"name": "output"})
        # create the module and store it
        self.module = ClassificationModule(inputlayers,
                                           hiddenlayers,
                                           outputlayer,
                                           self.featureinfo)
        # Decide on the lossfunction function here for training later!
        self.lossfunction = torch.nn.NLLLoss(ignore_index=-1)

    def init_sequencetagging(self, dataset):
        """Build the module for sequence tagging."""
        # NOTE: For sequence tagging, the shape of our input is slightly different:
        # - the indep is a list of features, as before
        # - but for each feature, there is a (padded) list of values
        # - each dep is also a padded list of values
        # In theory we could combine the features before going into the LSTM, or
        # we have different LSTMs for each feature and combine afterwards.
        # Here we combine before, so the output of e.g. a Linear layer is not just
        # a vector, but a matrix where one dimension is the batch, one dimension is the sequence
        # and one dimension is the value(vector). If we have batch size b, max sequence length s
        # and value dimension d, we should get shape b,s,d if batch_first is True, otherwise s,b,d
        n_classes = self.info["nClasses"]
        inputlayers = []
        # keep track of the number of input layer output dimensions
        inlayers_outdims = 0
        # if we have numeric features, create the numeric input layer
        if len(self.float_idxs) > 0:
            n_in = len(self.float_idxs)
            n_hidden = ModelWrapper.makeless(n_in, p1=0.5)
            lin = torch.nn.Linear(n_in, n_hidden)
            act = torch.nn.ELU()
            layer = torch.nn.Sequential(lin, act)
            inlayers_outdims += n_hidden
            lname = "input_numeric"
            inputlayers.append((layer, {"type": "numeric", "name": lname}))
            pass
        # if we have nominal features, create all the layers for those
        # TODO: may need to handle onehot features differently!!
        # remember which layers we already have for an embedding id
        nom_layers = {}
        for i in range(len(self.index_feats)):
            nom_feat = self.index_feats[i]
            nom_idx = self.index_idxs[i]
            vocab = nom_feat.vocab
            emb_id = vocab.emb_id
            if emb_id in nom_layers:
                emblayer = nom_layers.get(emb_id)
            else:
                emblayer = EmbeddingsModule(vocab)
                nom_layers[emb_id] = emblayer
            lname = "input_emb_%s_%s" % (i, emb_id)
            inputlayers.append((emblayer, {"type": "nominal", "name": lname}))
            inlayers_outdims += emblayer.emb_dims
        for i in range(len(self.indexlist_feats)):
            ngr_feat = self.indexlist_feats[i]
            nom_idx = self.indexlist_idxs[i]
            vocab = ngr_feat.vocab
            emb_id = vocab.emb_id
            if emb_id in nom_layers:
                emblayer = nom_layers.get(emb_id)
            else:
                emblayer = EmbeddingsModule(vocab)
                nom_layers[emb_id] = emblayer
            lname = "input_ngram_%s_%s" % (i, emb_id)
            ngramlayer = NgramModule(emblayer)
            inputlayers.append((ngramlayer, {"type": "ngram", "name": lname}))
            inlayers_outdims += ngramlayer.out_dim
        # Now create the hidden layers
        hiddenlayers = []


        # TODO: originally we always had this layer between the inputs and the LSTM, but
        # it may be better to just use a NOOP instead and just use the concatenated inputs.
        if False:
            n_hidden1lin_out = ModelWrapper.makeless(inlayers_outdims)
            hidden1lin = torch.nn.Linear(inlayers_outdims, n_hidden1lin_out)
            hidden1act = torch.nn.ELU()
            hidden1layer = torch.nn.Sequential(hidden1lin, hidden1act)
        else:
            n_hidden1lin_out = inlayers_outdims
            hidden1layer = None

        # for now, the size of the hidden layer is identical to the input size, up to 
        # a maximum of 200
        lstm_hidden_size = min(200, n_hidden1lin_out)
        lstm_bidirectional = False
        ## Now that we have combined the features, we create the lstm
        hidden2 = torch.nn.LSTM(input_size=n_hidden1lin_out,
                                  hidden_size=lstm_hidden_size,
                                  num_layers=1,
                                  # dropout=0.1,
                                  bidirectional=lstm_bidirectional,
                                  batch_first=True)
        # the outputs of the LSTM are of shape b, seq, hidden
        # We want to get softmax outputs for each, so we need to get this to
        # b, seq, nclasses

        # NOTE: we cannot use sequential here since the LSTM returns a tuple and
        # Sequential does not properly deal with this. So instead of adding the LSTM directly
        # we wrap it in a tiny custom wrapper that just returns the first element of the
        # tuple in the forward step
        hidden2 = TakeFromTuple(hidden2, which=0)

        # NOTE: if the LSTM is bidirectional, we need to double the size
        hidden3_size = lstm_hidden_size
        if lstm_bidirectional:
            hidden3_size *= 2
        hidden3 = torch.nn.Linear(hidden3_size, n_classes)
        if not hidden1layer:
            hidden = torch.nn.Sequential(hidden2, hidden3)
        else:
            hidden = torch.nn.Sequential(hidden1layer, hidden2, hidden3)
        hiddenlayers.append((hidden, {"name": "hidden"}))
        # Create the output layer
        out = torch.nn.LogSoftmax(dim=2)
        outputlayer = (out, {"name": "output"})
        # create the module and store it
        self.module = ClassificationModule(inputlayers,
                                           hiddenlayers,
                                           outputlayer,
                                           self.featureinfo)
        # For sequence tagging we cannot use CrossEntropyLoss
        self.lossfunction = torch.nn.NLLLoss(ignore_index=-1)



    def get_module(self):
        """Return the PyTorch module that has been built and is used by this wrapper."""
        return self.module

    def prepare_data(self, validationsize=None, file=None):
        """If file is not None, use the content of  the file, ignore the size.
        If validationsize is > 1, it is the absolute size, if < 1 it is the portion e.g. 0.01 to use."""
        # get the validation set
        if self.is_data_prepared:
            logger.warning("Called prepare_data after it was already called, doing nothing")
            return
        if file is not None:
            # use the file for validation
            self.dataset.split(convert=True, validation_file=file)
        else:
            if validationsize is not None:
                validationsize = float(validationsize)
            valsize = None
            valpart = None
            # TODO: allow not using a validation set at all!
            if validationsize is not None:
                if validationsize > 1 or validationsize == 0:
                    valsize = validationsize
                else:
                    valpart = validationsize
            else:
                valpart = 0.1
            self.dataset.split(convert=True, validation_part=valpart, validation_size=valsize)
        self.valset = self.dataset.validation_set_converted(as_batch=True)
        self.is_data_prepared = True
        # TODO if we have a validation set, calculate the class distribution here
        # this should be shown before training starts so the validation accuracy makes more sense
        # this can also be used to use a loss function that re-weights classes in case of class imbalance!

        # deps = self.valset[1]
        # TODO: calculate the class distribution but if sequences, ONLY for the non-padded parts of the sequences!!!!

    # TODO: this needs to use masking to undo the padding in the results!
    def apply(self, instancelist, converted=False, reshaped=False):
        """Given a list of instances in original format (or converted if converted=True), applies
        the model to them in evaluation mode and returns the following:
        As the first return value, the batch of predictions. This is a list of values (1 value for
        each instance in the batch) for classification and a list of lists (1 list representing a
        sequence for each instance in the batch) for sequence tagging.
        As the second value, returns the score/s for the returned predictions. This has the same
        shape as the first return value, but returns a score instead of each label.
        As the third value, returns a batch of confidence/scoring values. For classification,
        this is a list of lists, where the inner list is the label distribution. For sequence
        tagging, this is a list of list of lists, again with the label distribution as the inner-most
        list. Not that the mapping between the index of a value in the label distribution and
        the label itself can be figured out by the caller by retrieving the target vocab first.
        This may return additional data in the future or the format of what is returned may change.
        """
        batchsize = len(instancelist)
        if not converted:
            # TODO: check if and when to do instance normalization here!
            instancelist = [self.dataset.convert_indep(x) for x in instancelist]
            # logger.debug("apply: instances after conversion: %s" % (instancelist,))
        if not reshaped:
            instancelist = self.dataset.reshape_batch(instancelist, indep_only=True)
            # logger.debug("apply: instances after reshaping: %s" % (instancelist,))
        preds = self._apply_model(instancelist, train_mode=False)
        # logger.debug("apply: predictions result (shape %s): %s" % (preds.size(), preds,))
        # for now we only have classification (sequence/non-sequence) so
        # for this, we first use the torch max to find the most likely label index,
        # then convert back to the label itself. We also convert the torch probability vector
        # into a simple list of values
        ret = []
        nrClasses = self.dataset.nClasses
        if self.dataset.isSequence:
            # TODO: create a mask and return actual length sequences, not paddings from the tensor!
            # (not relevant in cases where the batchsize is only 1)
            # TODO: make it work for batchsize > 1!!!!!
            dims = preds.size()[-1]
            reshaped = preds.view(-1, dims).detach()
            # logger.debug("apply, reshaped=%s" % (reshaped,))
            reshaped = torch.exp(reshaped)
            # logger.debug("apply, reshaped-exp=%s" % (reshaped,))
            _, out_idxs = torch.max(reshaped, 1)
            # NOTE/IMPORTANT: we convert all numpy to list since numpy values (even just floats)
            # cannot get JSON serialized
            reshaped = reshaped.tolist()
            # predictions = out_idxs.cpu().numpy().tolist()
            predictions = out_idxs.tolist()
            probdists = [list(x) for x in reshaped]
            # logger.debug("apply, probdists=%s" % (probdists,))
            # logger.debug("apply, predictions=%s" % (predictions,))
            # logger.debug("apply, predictions type=%s" % (type(predictions),))
            # create the list of corresponding labels
            # TODO: again, this is a shortcut that only works if the batch has only one sequence
            logger.debug("len(predictions) %s" % (len(predictions),))
            #for i in range(len(predictions)):
            #    logger.debug("probdists[%s] %s" % (i, probdists[i],))
            #    logger.debug("predictions[%s] %s" % (i, predictions[i],))
            #    logger.debug("probdists[%s][predictions[%s]] %s" % (i, i, probdists[predictions[i]],))
            probs = [probdists[i][predictions[i]] for i in range(len(predictions))]
            labels = [self.dataset.target.idx2label(x) for x in predictions]
            # logger.debug("apply, labels=%s" % (labels,))
            # logger.debug("apply, probdists=%s" % (probdists,))
            # logger.debug("apply, probs=%s" % (probs,))
            # NOTE: currently the above code only works for a single instance and the
            # variables labels, probs, probdists are all for a single instance, not the batch.
            # So in order to make the result a batch, enclose each in a list as its single element
            return [labels], [probs], [probdists]
        else:
            # preds should be a 2d tensor of size batchsize x numberClasses
            assert len(preds.size()) == 2
            assert preds.size()[0] == batchsize
            assert preds.size()[1] == nrClasses
            probs, out_idxs = torch.max(preds, dim=1)
            probs = probs.detach().cpu().tolist()
            # out_idxs contains the class indices, need to convert back to labels
            getlabel = self.dataset.target.idx2label
            labels = [getlabel(x) for x in out_idxs]
            # for each instance in the batch return a list 
            # probs = [list(x) for x in preds]
            probdists = preds.detach().cpu().tolist()
            ret = labels, probs, probdists
        return ret

    def _apply_model(self, indeps, train_mode=False):
        """Apply the model to the list of indeps in the correct format for our Pytorch module
         and returns a list of predictions as Pytorch variables.
        train_mode influences if the underlying model is used in training mode or not.
        """
        if train_mode and not self.is_data_prepared:
            raise Exception("Must call train or prepare_data first")
        curmodeistrain = self.module.training
        if train_mode and not curmodeistrain:
            self.module.train()
            self.module.zero_grad()
        elif not train_mode and curmodeistrain:
            self.module.eval()

        output = self.module(indeps)
        # logger.debug("Output of model is of size %s: %s" % (output.size(), output, ))

        if self.module.training == curmodeistrain:
            self.module.train(curmodeistrain)
        return output

    def evaluate(self, validationinstances, train_mode=False, as_pytorch=True):
        """Apply the model to the independent part of the validationset instances and use the dependent part
        to evaluate the predictions. The validationinstances must be in batch format.
        Returns a tuple of loss and accuracy. By default this returns the loss as a
        pyTorch variable and accuracy as a pytorch tensor, if as_pytorch is set to False,
        returns floats instead.
        If prepared=True then validationinstances already contains everything as properly prepared PyTorch
        Variables.
        """
        if not self.is_data_prepared:
            raise Exception("Must call train or prepare_data first")
        # NOTE!!! the targets are what we get minus 1, which shifts the padding index to be -1
        # TODO: IF we use padded targets, we need to subtract 1 here, otherwise we have to leave this
        # as is!!
        targets = np.array(validationinstances[1])
        # v_deps = V(torch.LongTensor(targets), requires_grad=False)
        v_deps = torch.LongTensor(targets)
        if self._enable_cuda:
            v_deps = v_deps.cuda()
        v_preds = self._apply_model(validationinstances[0], train_mode=train_mode)
        # logger.debug("Got v_preds of size %s: %s" % (v_preds.size(), v_preds,))
        # logger.debug("Evaluating against targets of size %s: %s" % (v_deps.size(), v_deps))
        # TODO: not sure if and when to zero the grads for the loss function if we use it
        # in between training steps?
        # NOTE: the v_preds may or may not be sequences, if sequences we get the wrong shape here
        # so for now we simply put all the items (sequences and batch items) in the first dimension
        valuedim = v_preds.size()[-1]
        loss_function = self.lossfunction
        v_preds_reshape = v_preds.view(-1, valuedim)
        # !!DEBUG print("Predictions, reshaped, size=", v_preds_reshape.size(), "is", v_preds_reshape, file=sys.stderr)
        v_deps_reshape = v_deps.view(-1)
        # !!DEBUG print("Targets, reshaped, size=", v_deps_reshape.size(), "is", v_deps_reshape, file=sys.stderr)
        loss = loss_function(v_preds_reshape, v_deps_reshape)
        # calculate the accuracy as well, since we know we have a classification problem
        acc, correct, total = ModelWrapper.accuracy(v_preds, v_deps)
        # logger.debug("got loss %s accuracy %s" % (loss, acc, ))
        # print("loss=", loss, "preds=", v_preds, "targets=", v_deps, file=sys.stderr)
        # !!DEBUG sys.exit()
        if not as_pytorch:
            loss = float(loss)
            acc = float(acc)
        return tuple((loss, acc, correct, total))


    # the implementation should figure out best values if parameter
    # is set to None
    # Also, by default, the method should decide which format
    # to use for reading the data (original or converted)
    def train(self, max_epochs=20, batch_size=20, early_stopping=True, filenameprefix=None
              ):
        """Train the model on the dataset. max_epochs is the maximum number of
        epochs to train, but if early_stopping is enabled, it could be fewer.
        If early_stopping is True, then a default strategy is used.
        If set to a function that function (which must accept a standard set of parameters
        and return a boolean) is used.
        TODO: check if config should be used by default for the batch_size etc here!
        """

        # if this get set to True we bail out of all loops, save the model if necessary and stop training
        stop_it_already = False
        # this gets set by the signal handler and has the same effect as stop_it_already
        self.interrupted = False

        if early_stopping:
            if not filenameprefix:
                raise Exception("If early stopping is specified, filenameprefix is needed")
            if isinstance(early_stopping, bool):
                if early_stopping:
                    early_stopping_function = ModelWrapper.early_stopping_checker
                else:
                    early_stopping = lambda *args, **kwargs : False
            else:
                early_stopping_function = early_stopping
        if not self.is_data_prepared:
            logger.warning("Invoked train without calling prepare_data first, running default")
            self.prepare_data()
        # make sure we are in training mode
        self.module.train(mode=True)
        # set the random seed, every module must know how to handle this
        self.module.set_seed(self.random_seed)
        # the list of all validation losses so far
        validation_losses = []
        # list of all validation accuracies so far
        validation_accs = []
        # total number of batches processed over all epochs
        totalbatches = 0
        # for calculating loss and acc over a number of batches or instances for reporting
        report_correct = 0
        report_total = 0
        report_loss = 0
        # best validation accuracy so far
        # initialize the last epoch number for validation to 1 so we do not validate right away
        last_epoch = 1
        best_acc = 0.0
        saved_model_name = None
        for epoch in range(1, max_epochs+1):
            # batch number within an epoch
            batch_nr = 0
            # number of instances already used for training during this epoch
            nr_instances = 0
            # for calculating loss and acc over the whole epoch / training set
            epoch_correct = 0
            epoch_total = 0
            epoch_loss = 0
            for batch in self.dataset.batches_converted(train=True, batch_size=batch_size):
                batch_nr += 1
                totalbatches += 1
                nr_instances += batch_size  # we should use the actual batch size which could be less
                self.module.zero_grad()
                # import ipdb
                # ipdb.set_trace()
                (loss, acc, correct, total) = self.evaluate(batch, train_mode=True)
                logger.debug("Epoch=%s, batch=%s: TRAINING loss=%s acc=%s" %
                             (epoch, batch_nr, f(loss), f(acc)))
                loss.backward()
                report_loss += float(loss)
                report_correct += float(correct)
                report_total += float(total)
                epoch_loss += float(loss)
                epoch_correct += float(correct)
                epoch_total += float(total)
                self.optimizer.step()
                # evaluation on the training set only for reporting
                if (self.report_every_batches and ((totalbatches % self.report_every_batches) == 0)) or \
                        (self.report_every_instances and ((nr_instances % self.report_every_instances) == 0)):
                    logger.info("Epoch=%s, batch=%s, insts=%s: loss=%s acc=%s / epoch_loss=%s epoch_acc=%s" %
                                (epoch, batch_nr, nr_instances,
                                 f(report_loss), f(report_correct / report_total),
                                 f(epoch_loss), f(epoch_correct / epoch_total)))
                    report_loss = 0
                    report_correct = 0
                    report_total = 0
                # this is for validating against the validation set and possibly early stopping
                if (self.validate_every_batches and ((totalbatches % self.validate_every_batches) == 0)) or\
                        (self.validate_every_epochs and ((epoch - last_epoch) == self.validate_every_epochs)) or \
                        (self.validate_every_instances and ((nr_instances % self.validate_every_instances) == 0)):
                    # evaluate on validation set
                    last_epoch = epoch
                    (loss_val, acc_val, correct, total) = self.evaluate(self.valset, train_mode=False)
                    logger.info("Epoch=%s, VALIDATION: loss=%s acc=%s" %
                                (epoch, f(loss_val), f(acc_val)))
                    validation_losses.append(float(loss_val))
                    validation_accs.append(float(acc_val))
                    # if we have early stopping, check if we should stop
                    if early_stopping:
                        stop_it_already = early_stopping_function(
                            losses=validation_losses, accs=validation_accs, metric=self.config.get("es_metric", "loss"))
                        if stop_it_already:
                            logger.info("Early stopping criterion reached, stopping training, best validation acc: %s" %
                                        (best_acc,))
                    # if the current validation accuracy is better than what we had so far, save
                    # the model
                    if acc_val > best_acc:
                        best_acc = acc_val
                        saved_model_name = self.save_model(filenameprefix)
                        self.best_model_saved = True

                if self.stopfile and os.path.exists(self.stopfile):
                    logger.info("Stop file found, removing and terminating training, best validation acc: %s" %
                                (best_acc,))
                    os.remove(self.stopfile)
                    stop_it_already = True
                if stop_it_already or self.interrupted:
                    break
            if stop_it_already or self.interrupted:
                self.interrupted = False
                break
        logger.info("Training completed, best validation acc={}, model saved to {}".format(best_acc, saved_model_name))

    def checkpoint(self, filenameprefix, checkpointnr=None):
        """Save the module, adding a checkpoint number in the name."""
        # TODO: eventually this should get moved into the module?
        cp = checkpointnr
        if cp is None:
            cp = self.checkpointnr
            self.checkpointnr += 1
        torch.save(self.module, filenameprefix + ".module.pytorch")

    def save_model(self, filenameprefix):
        start = timeit.timeit()
        filename = filenameprefix + ".module.pytorch"
        torch.save(self.module, filename)
        end = timeit.timeit()
        logger.info("Saved model to %s in %s" % (filename, f(abs(end - start))))
        return filename

    def save(self, filenameprefix):
        # store everything using pickle, but we do not store the module or the dataset
        # the dataset will simply get recreated when loading, but the module needs to get saved
        # separately

        # only if we did not already save the best model during training for some reason
        if not self.best_model_saved:
            self.save_model(filenameprefix)
        assert hasattr(self, 'metafile')
        filename = filenameprefix+".wrapper.pickle"
        with open(filename, "wb") as outf:
            start = timeit.timeit()
            pickle.dump(self, outf)
            end = timeit.timeit()
            logger.info("Saved wrapper to %s in %s" % (filename, f(abs((end-start)))))


    def init_after_load(self, filenameprefix, cuda=None):
        """
        If cuda is not None, try to load the module directly to cpu or cuda, as requested.
        If cuda is None, let pytorch decide what to do.
        """
        self.dataset = Dataset(self.metafile)
        self.init_from_dataset()
        if cuda is None:
            self.module = torch.load(filenameprefix+".module.pytorch")
        else:
            if cuda:
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            self.module = torch.load(filenameprefix + ".module.pytorch", map_location=str(device))
            # make doubly sure
            self.module.to(device)
            self.set_cuda(cuda)
        self.is_data_prepared = False
        self.valset = None

    def __getstate__(self):
        """Currently we do not pickle the dataset instance but rather re-create it when loading,
        and we do not pickle the actual pytorch module but rather use the pytorch-specific saving
        and loading mechanism."""
        # print("DEBUG: self keys=", self.__dict__.keys(), file=sys.stderr)
        assert hasattr(self, 'metafile')
        state = self.__dict__.copy()  # this creates a shallow copy
        # print("DEBUG: copy keys=", state.keys(), file=sys.stderr)
        assert 'metafile' in state
        # do not save these transient variables:
        del state['dataset']
        del state['module']
        del state['valset']
        del state['optimizer']
        del state['is_data_prepared']
        # in case we need to check if any other variable is from package torch:
        # for k, v in state.items():
        #     print('DEBUG: variable {} from package {}'.format(k, type(v).__module__))
        #     if type(v).__module__ == "torch":
        #         print("DEBUG: !!! {} is a torch variable!".format(k))
        print("DEBUG: metafile is {}".format(state['metafile']))
        return state

    def __setstate__(self, state):
        """We simply restore everything that was pickled earlier, the missing fields
        then need to get restored using the _init_after_load method (called from load)"""
        assert 'metafile' in state
        self.__dict__.update(state)
        assert hasattr(self, 'metafile')

    def __repr__(self):
        repr = "ModelWrapperSimple(config=%r, cuda=%s):\nmodule=%s\noptimizer=%s\nlossfun=%s" % \
             (self.config, self._enable_cuda, self.module, self.optimizer, self.lossfunction)
        return repr
