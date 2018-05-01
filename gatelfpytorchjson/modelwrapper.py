import math
import numpy as np
import torch

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
    def accuracy(batch_predictions, batch_targets, targets_onehot=False):
        """Calculate accuracy for the predictions (one-hot vectors) based on the true class indices in batch_targets.
        Targets are assumed to be indices unless targets_onehot is True.
        Both parameters are assumed to be torch Variables."""
        # NOTE: in case we have sequences we reshape to put everything that is not the last dimension
        # into the first dimension for the predictions
        dims = batch_predictions.size()[-1]
        _, out_idxs = torch.max(batch_predictions.data.view(-1, dims), 1)
        n_correct = int(out_idxs.eq(batch_targets.data.view(-1)).sum())
        acc = n_correct / float(batch_targets.data.view(-1).size()[0])
        return acc
