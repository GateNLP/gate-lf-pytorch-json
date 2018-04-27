from . modelwrapper import ModelWrapper
from . embeddingsmodule import EmbeddingsModule
from . ngrammodule import NgramModule
import torch
import torch.nn
import torch.optim
import math
from torch.autograd import Variable as V
import torch.nn.functional as F
from .classificationmodelsimple import ClassificationModelSimple
from .takefromtuple import TakeFromTuple
import logging
import sys
import statistics
import pickle
from gatelfdata import Dataset

# Basic usage:
# ds = Dataset(metafile)
# wrapper = ModelWrapperSimple(ds) # or some other subclass
# wrapper.train()
# # get some data for application some where
# instances = get_them()
# preditions = wrapper.apply(instances)
# NOTE: maybe use the same naming conventions as scikit learn here!!


class ModelWrapperSimple(ModelWrapper):

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
    def __init__(self, dataset, config=None, cuda=None):
        """This requires a gatelfdata Dataset instance and can optionally take a dictionary with
        configuration/initialization options (NOT SUPPORTED YET).
        If cuda is None, then if cuda is available it will be used. True and False
        require and prohibit the use of cuda unconditionally.
        """
        super().__init__(dataset)
        self.cuda = cuda
        cuda_is_available = torch.cuda.is_available()
        if self.cuda is None:
            enable_cuda = cuda_is_available
        else:
            enable_cuda = self.cuda
        self._enable_cuda = enable_cuda  # this tells us if we should actually set cuda or not
        self.dataset = dataset
        self.init_from_dataset()
        # various configuration settings which can be set before passing on control to the
        # task-speicific initialization
        self.validate_every_batches = 100
        self.validate_every_epochs = None
        self.is_data_prepared = False
        self.valset = None   # Validation set created by prepare_data
        self.lossfunction = None
        self.module = None  # the init_<TASK> method actually sets this!!
        if self.info["isSequence"]:
            self.init_sequencetagging(dataset)
        else:
            if self.info["targetType"] == "nominal":
                self.init_classification(dataset)
            else:
                raise Exception("Target type not yet implemented: %s" % self.info["targetType"])
        # self.optimizer = torch.optim.SGD(self.module.parameters(), lr=0.001, momentum=0.9)
        # self.optimizer = torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
        # self.optimizer = torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
        self.optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0 )
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


    def init_classification(self, dataset):
        n_classes = self.info["nClasses"]
        inputlayers = []
        # keep track of the number of input layer output dimensions
        inlayers_outdims = 0
        # if we have numeric features, create the numeric input layer
        if len(self.float_idxs) > 0:
            n_in = len(self.float_idxs)
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
        # Decide on the lossfunction function here for training later!
        self.lossfunction = torch.nn.CrossEntropyLoss()
        if self._enable_cuda:
            self.module.cuda()
            self.lossfunction.cuda()

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
            hidden1drp = torch.nn.Dropout(p=0.2)
            hidden1act = torch.nn.ELU()
            hidden1layer = torch.nn.Sequential(hidden1lin, hidden1drp, hidden1act)
        else:
            n_hidden1lin_out = inlayers_outdims
            hidden1layer = None

        # for now, the size of the hidden layer is identical to the input size, up to 
        # a maximum of 200
        lstm_hidden_size = min(200,n_hidden1lin_out)
        lstm_bidirectional = False
        ## Now that we have combined the features, we create the lstm
        hidden2 = torch.nn.LSTM(input_size=n_hidden1lin_out,
                                  hidden_size=lstm_hidden_size,
                                  num_layers=1,
                                  dropout=0.1,
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
        out = torch.nn.LogSoftmax(dim=1)
        outputlayer = (out, {"name": "output"})
        # create the module and store it
        self.module = ClassificationModelSimple(inputlayers,
                                                hiddenlayers,
                                                outputlayer,
                                                self.featureinfo)
        # For sequence tagging we cannot use CrossEntropyLoss
        self.lossfunction = torch.nn.NLLLoss(ignore_index=0)
        if self._enable_cuda:
            self.module.cuda()
            self.lossfunction.cuda()



    def get_module(self):
        """Return the PyTorch module that has been built and is used by this wrapper."""
        return self.module

    def prepare_data(self, validationsize=None):
        # get the validation set
        if self.is_data_prepared:
            return
        valsize = None
        valpart = 0.1
        # TODO: allow not using a validation set at all!
        if validationsize:
            if isinstance(validationsize, int):
                valsize = validationsize
            elif isinstance(validationsize, float):
                valpart = validationsize
            else:
                raise Exception("Parameter validationsize must be a float or int")
        self.dataset.split(convert=True, validation_part=valpart, validation_size=valsize)
        self.valset = self.dataset.validation_set_converted(as_batch=True)
        self.is_data_prepared = True
        # if we have a validation set, calculate the class distribution here 
        # this should be shown before training starts so the validation accuracy makes more sense
        # this can also be used to use a loss function that re-weights classes in case of class imbalance!
        deps = self.valset[1]
        # TODO: calculate the class distribution but if sequences, ONLY for the non-padded parts of the sequences!!!!

    def apply(self, instancelist, converted=False, reshaped=False):
        """Given a list of instances in original format (or converted if converted=True), applies
        the model to them in evaluation mode and returns the predictions in the following format as a list
        with the following elements:
        First, a list of the predicted labels, label sequences or values, as many elements as there are instances.
        Second, a list of additional values for each of the labels, whith a list of values corresponding
        to each of the labels or values in the first list. For labels, this is the probablity distribution
        over all labels (TODO: how to index this??)
        for values this could be confidence intervals, variances etc. That second element is optional.
        """
        batchsize = len(instancelist)
        if not converted:
            # TODO: check if and when to do instance normalization here!
            instancelist = [ self.dataset.convert_indep(x) for x in instancelist]
            print("\nDEBUG: instances after conversion: ", instancelist, file=sys.stderr)
        if not reshaped:
            instancelist = self.dataset.reshape_batch(instancelist, indep_only=True)
            print("\nDEBUG: instances after reshaping: ", instancelist, file=sys.stderr)
        preds = self._apply_model(instancelist, train_mode=False)
        # for now we only have classification (sequence/non-sequence) so
        # for this, we first use the torch max to find the most likely label index,
        # then convert back to the label itself. We also convert the torch probability vector
        # into a simple list of values
        ret = []
        nrClasses = self.dataset.nClasses
        if self.dataset.isSequence:
            raise Exception("Apply not yet implemented for sequences")
        else:
            # preds should be a 2d tensor of size batchsize x numberClasses
            assert len(preds.size()) == 2
            assert preds.size()[0] == batchsize
            assert preds.size()[1] == nrClasses
            _, out_idxs = torch.max(preds.data, dim=1)
            # out_idxs contains the class indices, need to convert back to labels
            getlabel = self.dataset.target.idx2label
            labels = [getlabel(x) for x in out_idxs]
            probs = [list(x) for x in preds.data]
            ret = [labels, probs]
        return ret

    def _apply_model(self, indeps, train_mode=False):
        """Apply the model to the list of indeps in the correct format for our Pytorch module
         and returns a list of predictions as Pytorch variables.
        train_mode influences if the underlying model is used in training mode or not.
        """
        if not self.is_data_prepared:
            raise Exception("Must call train or prepare_data first")
        curmodeistrain = self.module.training
        if train_mode and not curmodeistrain:
            self.module.train()
            self.module.zero_grad()
        elif not train_mode and curmodeistrain:
            self.module.eval()

        output = self.module(indeps)

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
        v_deps = V(torch.LongTensor(validationinstances[1]), requires_grad=False)
        if self._enable_cuda:
            v_deps = v_deps.cuda()
        v_preds = self._apply_model(validationinstances[0], train_mode=train_mode)
        # TODO: not sure if and when to zero the grads for the loss function if we use it
        # in between training steps?
        # NOTE: the v_preds may or may not be sequences, if sequences we get the wrong shape here
        # so for now we simply put all the items (sequences and batch items) in the first dimension
        valuedim = v_preds.size()[-1]
        loss = self.lossfunction(v_preds.view(-1, valuedim), v_deps.view(-1))
        # calculate the accuracy as well, since we know we have a classification problem
        acc = ModelWrapper.accuracy(v_preds, v_deps)
        if not as_pytorch:
            loss = float(loss)
            acc = float(acc)
        return tuple((loss, acc))

    # the implementation should figure out best values if parameter
    # is set to None
    # Also, by default, the method should decide which format
    # to use for reading the data (original or converted)
    def train(self, max_epochs=None, batch_size=None, validationsize=None, early_stopping=True,
              ):
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
        # get the validation set
        self.prepare_data()
        # make sure we are in training mode
        self.module.train(mode=True)
        if not batch_size:
            batch_size = 10
        # val_indeps = self.valset[0]
        # val_targets = V(torch.LongTensor(self.valset[1]), requires_grad=False)
        stop_it_already = False
        validation_losses = []
        print("DEBUG: before running training...", file=sys.stderr)
        totalbatches = 0
        last_accs = []
        last_losses = []
        for epoch in range(1, max_epochs+1):
            batch_nr = 0
            for batch in self.dataset.batches_converted(train=True, batch_size=batch_size):
                batch_nr += 1
                totalbatches += 1
                self.module.zero_grad()
                (loss, acc) = self.evaluate(batch, train_mode=True)
                # logger.debug("Batch lossfunction/acc for epoch=%s, batch=%s: %s / %s" %
                #             (epoch, batch_nr, float(loss), acc))
                # print("Batch lossfunction/acc for epoch=%s, batch=%s: %s / %s" % (epoch, batch_nr,
                # float(lossfunction), acc), file=sys.stderr)
                loss.backward()
                self.optimizer.step()
                last_accs.append(float(acc))
                last_losses.append(float(loss))
                if (self.validate_every_batches and ((totalbatches % self.validate_every_batches) == 0)) or\
                        (self.validate_every_epochs and ((epoch % self.validate_every_epochs) == 0)):
                    # evaluate on validation set
                    (loss_val, acc_val) = self.evaluate(self.valset, train_mode=False)
                    # self.module.eval()
                    # out_val = self.module(val_indeps)
                    # loss_val = self.lossfunction(out_val, val_targets)
                    # acc_val = ModelWrapper.accuracy(out_val, val_targets)
                    validation_losses.append(float(loss_val))
                    # if we have early stopping, check if we should stop
                    var = None
                    if early_stopping:
                        (stop_it_already, var) = early_stopping_function(validation_losses)
                        if stop_it_already:
                            logger.info("Early stopping criterion reached, stopping training, var=%s" % var)
                    avg_tloss = statistics.mean(last_losses)
                    avg_tacc = statistics.mean(last_accs)
                    var_vloss = None
                    if len(validation_losses) > 10:
                        var_vloss = statistics.variance(validation_losses[-10:])
                    last_losses = []
                    last_accs = []
                    logger.info("EVAL e=%s,b=%s,tloss/vloss/"
                                "vloss-var/tacc/vacc: %s / %s / %s / %s / %s" %
                                (epoch, batch_nr, avg_tloss, float(loss_val), var_vloss, avg_tacc, acc_val))
                if stop_it_already:
                    break
            if stop_it_already:
                break

    def save(self, filenameprefix):
        # store everything using pickle, but we do not store the module or the dataset
        # the dataset will simply get recreated when loading, but the module needs to get saved
        # separately

        # TODO: eventually, make every module know what is the best way to save and load itself,
        # and delegate, but for now we just use the standard pytorch approach
        # self.module.save(self.module, filenameprefix+"module.pytorch")
        torch.save(self.module, filenameprefix+".module.pytorch")
        assert hasattr(self, 'metafile')
        with open(filenameprefix+".wrapper.pickle", "wb") as outf:
            pickle.dump(self, outf)

    @classmethod
    def load(cls, filenameprefix):
        with open(filenameprefix+".wrapper.pickle", "rb") as inf:
            w = pickle.load(inf)
        print("DEBUG: restored instance keys=", w.__dict__.keys(), file=sys.stderr)
        assert hasattr(w, 'metafile')
        w.dataset = Dataset(w.metafile)
        w.init_from_dataset()
        w.module = torch.load(filenameprefix+".module.pytorch")
        return w

    def __getstate__(self):
        """Currently we do not pickle the dataset instance but rather re-create it when loading,
        and we do not pickle the actual pytorch module but rather use the pytorch-specific saving
        and loading mechanism."""
        print("DEBUG: self keys=", self.__dict__.keys(), file=sys.stderr)
        assert hasattr(self, 'metafile')
        state = self.__dict__.copy()  # this creates a shallow copy
        print("DEBUG: copy keys=", state.keys(), file=sys.stderr)
        assert 'metafile' in state
        del state['dataset']
        del state['module']
        return state

    def __setstate__(self, state):
        """We simply restore everything that was pickled earlier plus manually rebuild the dataset
        instance and manually restore the pytorch module."""
        assert 'metafile' in state
        self.__dict__.update(state)
        assert hasattr(self, 'metafile')

