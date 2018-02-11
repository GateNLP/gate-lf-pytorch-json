from gatelfpytorch.modelwrappersimple import ModelWrapperSimple
from gatelfdata import Dataset
import unittest
import os
import sys
import logging

logger = logging.getLogger("gatelfpytorch")
logger.setLevel(logging.ERROR)
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


class Test1(unittest.TestCase):

    def test1_1(self):
        ds = Dataset(TESTFILE1)
        wrapper = ModelWrapperSimple(ds)
        m = wrapper.get_module()
        print("\nDEBUG: module:", m, file=sys.stderr)
        wrapper.train()
        pass
