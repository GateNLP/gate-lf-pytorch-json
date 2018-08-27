import math
import numpy as np
import torch
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)



# NOTE/TODO: eventually we will refactor some generally useful stuff from the
# subclasses up to here
class ModelWrapper(object):

    # every subclass should implement these methods:
    def __init__(self, dataset, config={}):
        pass

    def train(self, epochs=None, batchsize=None):
        # train the model
        pass

    def save(self):
        # save the wrapper and contained pytorch model
        pass

    @classmethod
    def load(cls):
        # return an instance of the wrapper ready for application
        # or maybe continuing training
        pass

    def apply(self, indeps):
        # return output for the indeps
        pass

    # Additional useful methods
    @staticmethod
    def early_stopping_checker(evaluations, k=10, max_variance=0.0001):
        """Takes an array of floats, with the most recent evaluation being the last in the list
        and returns True if training should be stopped"""
        # simple criterion: if the variance of the last k evaluation is smaller than
        # the max_variance, stop.
        if len(evaluations) > k:
            var = np.var(evaluations[-k:])
            if var < max_variance:
                return tuple((True, var))
        else:
            var = None
        return tuple((False, var))

    @staticmethod
    def makeless(n, func=math.pow, preshift=-1.0, postshift=1.0, p1=0.5):
        val = int(func((n+preshift), p1)+postshift)
        return val

    @staticmethod
    def accuracy(batch_predictions, batch_targets, pad_index=-1):
        """Calculate accuracy for the predictions (one-hot vectors) based on the true class indices in batch_targets.
        Targets are assumed to be indices. Both parameters are assumed to be torch Variables.
        """

        # NOTE: for classification the shapes should be:
        # * batch_predictions: batch_size, n_classes
        # * batch_targets: batch_size
        # For sequence tagging the shapes should be:
        # * batch_predictions: batch_size, max_seq_len, n_classes
        # * batch_targets: batch_size, max_seq_len

        n_pred_dims = len(batch_predictions.size())
        pred_size = batch_predictions.size()[-1]  # the size of the predictions dimension

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
        acc = n_correct / np.sum(mask)
        logger.debug("Total=%s, correct=%s, acc=%s" % (np.sum(mask), n_correct, acc,))
        # import ipdb
        # ipdb.set_trace()
        return acc
