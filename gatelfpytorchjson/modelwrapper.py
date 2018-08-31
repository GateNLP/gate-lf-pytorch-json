import math
import numpy as np
import torch
import logging
import sys
import pickle

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)


class ModelWrapper(object):
    """Common base class for all wrappers. Defines instance methods which are the same
    for all subclasses plus common static utility methods."""

    # This has been defined so that subclasses can call the superclass init function
    # with parameter config. This does nothing yet, but could do some common initialization
    # processing in the future.
    def __init__(self, dataset, config={}):
        pass


    @classmethod
    def load(cls, filenameprefix):
        """Load a saved wrapper instance and return it. The file name is made of the
        filenameprefix with '.wrapper.pickle' appended."""
        with open(filenameprefix+".wrapper.pickle", "rb") as inf:
            w = pickle.load(inf)
        logger.debug("Restored instance keys=%s" % (w.__dict__.keys(),))
        assert hasattr(w, 'metafile')
        w.init_after_load(filenameprefix)
        return w

    # Useful utility methods below this line

    @staticmethod
    def early_stopping_checker(losses=None, accs=None):
        """Takes two lists of numbers, representing the losses and accuracies of all validation
        steps. If the accs are not null, early stopping is initiated if the accuracy of the last
        2 validations are not better then the one before. If accuracies are not specified
        then this method always returns false, meaning no early stopping. The losses are ignored
        by this method"""
        if not accs:
            return False
        if len(accs) < 3:
            return False
        if accs[-1] < accs[-3] and accs[-2] < accs[-3]:
            return True

    @staticmethod
    def makeless(n, func=math.pow, preshift=-1.0, postshift=1.0, p1=0.5):
        """Function to return logarithmic or inverse polynomial values for such tasks
        as calculating number of dimensions from vocabulary size."""
        val = int(func((n+preshift), p1)+postshift)
        return val

    @staticmethod
    def accuracy(batch_predictions, batch_targets, pad_index=-1):
        """Calculate the accuracy from a tensor with predictions, which contains scores for each
        class in the last dimension (higher scores are better) and a tensor with target indices.
        Tensor elements where the target has the padding index are ignored.
        If the tensors represent sequences the shape of the predictions is batchsize, maxseqlen, nclasses
        and of the targets is batchsize, maxseqlen, otherwise the predictions have shape
        batchsize, nclasses, targets have shape batchsize
        """

        # n_pred_dims = len(batch_predictions.size())  # this should be 3 for sequences, otherwise 2
        pred_size = batch_predictions.size()[-1]  # the size of the predictions dimension = nclasses

        # first reshape so that we have the prediction scores in the last/second dimension
        # then find the argmax index along the last/second dimension (dimension=1)
        _, out_idxs = torch.max(batch_predictions.data.view(-1, pred_size), 1)


        # TODO: it may be more efficient to calculate the accuracy differently for sequences and for
        # classification and avoid using numpy here
        # Instead we could use, just with torch tensors:
        # mask = (targets != -1)
        # same = (targets == predictions)
        # vals = torch.masked_select(same, mask)
        # total = vals.size()[0]
        # correct = vals.sum()

        targets = batch_targets.data.view(-1).cpu().numpy()
        logger.debug("targets reshaped: %s" % (targets,))
        pred_idxs = out_idxs.cpu().numpy()
        logger.debug("pred_idxs reshaped: %s" % (pred_idxs,))
        mask = (targets != pad_index)
        logger.debug("mask reshaped: %s" % (mask,))
        n_correct = np.sum(pred_idxs[mask] == targets[mask])
        n_total = np.sum(mask)
        acc = n_correct / n_total
        logger.debug("Total=%s, correct=%s, acc=%s" % (np.sum(mask), n_correct, acc,))
        # import ipdb
        # ipdb.set_trace()
        return acc, n_correct, n_total
