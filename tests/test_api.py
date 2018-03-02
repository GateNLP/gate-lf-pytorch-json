from gatelfpytorch.modelwrappersimple import ModelWrapperSimple
# from gatelfpytorch.modelwrapper import ModelWrapper
from gatelfdata import Dataset
import unittest
import os
import sys
import logging
import torch
# from torch.autograd import Variable as V

logger = logging.getLogger("gatelfdata")
logger.setLevel(logging.ERROR)
logger = logging.getLogger("gatelfpytorch")
logger.setLevel(logging.DEBUG)
streamhandler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)
filehandler = logging.FileHandler("test_api.log")
logger.addHandler(filehandler)

TESTDIR = os.path.join(os.path.dirname(__file__), '.')
DATADIR = os.path.join(TESTDIR, 'data')
print("DEBUG: datadir is ", TESTDIR, file=sys.stderr)

TESTFILE1 = os.path.join(DATADIR, "class-ionosphere.meta.json")
TESTFILE2 = os.path.join(DATADIR, "class-ngram-sp1.meta.json")
TESTFILE3 = os.path.join(DATADIR, "class-window-pos1.meta.json")
TESTFILE4 = os.path.join(DATADIR, "seq-pos1.meta.json")

# TODO: set to true if we have cuda
if torch.cuda.device_count() > 0:
    SLOW_TESTS = True
else:
    SLOW_TESTS = False

# In case we want to temporarily override
SLOW_TESTS = True

class Test1(unittest.TestCase):

    # def test1_1(self):
    #     ds = Dataset(TESTFILE1)
    #     torch.manual_seed(1)  # make results based on random weights repeatable
    #     wrapper = ModelWrapperSimple(ds)
    #     print("\nDEBUG: dataset=", wrapper.dataset, file=sys.stderr)
    #     m = wrapper.get_module()
    #     wrapper.prepare_data()
    #     print("\nDEBUG: module:", m, file=sys.stderr)
    #     (loss, acc) = wrapper.evaluate(wrapper.valset, train_mode=False, as_pytorch=False)
    #     assert acc < 0.7
    #     print("\nDEBUG: test1_1 before training loss/acc=%s/%s" % (loss, acc), file=sys.stderr)
    #     if SLOW_TESTS:
    #         wrapper.train(batch_size=20, max_epochs=250, early_stopping=False)
    #         (loss, acc) = wrapper.evaluate(wrapper.valset, train_mode=False, as_pytorch=False)
    #         assert acc > 0.7
    #         print("\nDEBUG: test1_1 after training loss/acc=%s/%s" % (loss, acc), file=sys.stderr)
    #
    # def test1_2(self):
    #     ds = Dataset(TESTFILE2)
    #     torch.manual_seed(1)  # make results based on random weights repeatable
    #     wrapper = ModelWrapperSimple(ds)
    #     print("\nDEBUG: dataset=", wrapper.dataset, file=sys.stderr)
    #     wrapper.prepare_data()
    #     m = wrapper.get_module()
    #     print("\nDEBUG: module:", m, file=sys.stderr)
    #     wrapper.validate_every_batches = 10
    #     # wrapper.train(batch_size=33,
    #     # early_stopping=lambda x: ModelWrapper.early_stopping_checker(x, max_variance=0.0000001))
    #     (loss, acc) = wrapper.evaluate(wrapper.valset, train_mode=False, as_pytorch=False)
    #     print("\nDEBUG: test1_2 before training loss/acc=%s/%s" % (loss, acc), file=sys.stderr)
    #     assert acc < 0.55
    #     wrapper.optimizer = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.0)
    #     # NOTE: so far this is very slow and does not learn on the tiny set so we permanently deactivate,
    #     # not just if we want to avoid slow tests
    #     if SLOW_TESTS and False:
    #         wrapper.train(batch_size=5, max_epochs=6, early_stopping=False)
    #         (loss, acc) = wrapper.evaluate(wrapper.valset, train_mode=False)
    #         assert acc > 0.6
    #         print("\nDEBUG: test1_1 after training loss/acc=%s/%s" % (loss, acc), file=sys.stderr)
    #
    # def test1_3(self):
    #     ds = Dataset(TESTFILE3)
    #     torch.manual_seed(1)  # make results based on random weights repeatable
    #     wrapper = ModelWrapperSimple(ds)
    #     print("\nDEBUG: dataset=", wrapper.dataset, file=sys.stderr)
    #     m = wrapper.get_module()
    #     print("\nDEBUG: module:", m, file=sys.stderr)
    #     wrapper.validate_every_batches = 10
    #     wrapper.prepare_data()
    #     (loss, acc) = wrapper.evaluate(wrapper.valset, train_mode=False, as_pytorch=False)
    #     print("\nDEBUG: test1_3 before training loss/acc=%s/%s" % (loss, acc), file=sys.stderr)
    #     assert acc < 0.05
    #     if SLOW_TESTS:
    #         wrapper.train(batch_size=20, max_epochs=30, early_stopping=False)
    #         (loss, acc) = wrapper.evaluate(wrapper.valset, train_mode=False, as_pytorch=False)
    #         print("\nDEBUG: test1_3 after training loss/acc=%s/%s" % (loss, acc), file=sys.stderr)
    #         assert acc > 0.3
    #
    # def test1_4(self):
    #     ds = Dataset(TESTFILE4)
    #     torch.manual_seed(1)  # make results based on random weights repeatable
    #     wrapper = ModelWrapperSimple(ds)
    #     print("\nDEBUG: dataset=", wrapper.dataset, file=sys.stderr)
    #     m = wrapper.get_module()
    #     print("\nDEBUG: module:", m, file=sys.stderr)
    #     wrapper.validate_every_batches = 10
    #     wrapper.prepare_data()
    #     (loss, acc) = wrapper.evaluate(wrapper.valset, train_mode=False, as_pytorch=False)
    #     print("\nDEBUG: test1_4 before training loss/acc=%s/%s" % (loss, acc), file=sys.stderr)
    #     assert acc < 0.2134
    #     if SLOW_TESTS:
    #         wrapper.train(batch_size=20, max_epochs=30, early_stopping=False)
    #         (loss, acc) = wrapper.evaluate(wrapper.valset, train_mode=False, as_pytorch=False)
    #         print("\nDEBUG: test1_4 after training loss/acc=%s/%s" % (loss, acc), file=sys.stderr)
    #         assert acc > 0.215

    def test1_5(self):
        """Test saving and restoring a model"""
        ds = Dataset(TESTFILE1)
        ds.split()
        torch.manual_seed(1)  # make results based on random weights repeatable
        wrapper = ModelWrapperSimple(ds)
        print("DEBUG: wrapper.dataset=", wrapper.dataset, file=sys.stderr)
        print("DEBUG: wrapper.metafile=", wrapper.metafile, file=sys.stderr)
        assert hasattr(wrapper, 'metafile')
        m = wrapper.get_module()
        wrapper.prepare_data()
        print("\nDEBUG: module:", m, file=sys.stderr)
        valset = wrapper.valset
        (loss, acc) = wrapper.evaluate(valset, train_mode=False, as_pytorch=False)
        print("\nDEBUG: test1_5 before training loss/acc=%s/%s" % (loss, acc), file=sys.stderr)
        assert acc < 0.7
        wrapper.train(batch_size=20, max_epochs=150, early_stopping=False)
        (loss, acc) = wrapper.evaluate(valset, train_mode=False, as_pytorch=False)
        print("\nDEBUG: test1_5 after training loss/acc=%s/%s" % (loss, acc), file=sys.stderr)
        assert acc > 0.7
        # save the model
        assert hasattr(wrapper, 'metafile')
        wrapper.save("t1_5")
        wrapper = None
        m = None
        wrapper2 = ModelWrapperSimple.load("t1_5")
        ds2 = wrapper2.dataset
        (loss, acc) = wrapper2.evaluate(valset, train_mode=False, as_pytorch=False)
        print("\nDEBUG: test1_5 after restoring loss/acc=%s/%s" % (loss, acc), file=sys.stderr)
        assert acc > 0.7

        # No test application with the restored model
        vals_orig = ds.validation_set_orig()
        indeps_orig = vals_orig[0]
        indeps_orig_0 = [indeps_orig[0]]
        print("\nDEBUG: 1-instance batch: ", indeps_orig_0, file=sys.stderr)
        preds = wrapper2.apply(indeps_orig_0)
        print("\nDEBUG: prediction for 1-instance batch: ", preds,  file=sys.stderr)
        # first element in preds is the list of labels, second one the list of of probabilities
        labels = preds[0]
        probs = preds[1]
        assert labels[0] == 'g'
