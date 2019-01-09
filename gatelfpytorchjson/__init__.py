# check minimum requirements: python 3 and minimum torch version
import sys
import torch
__version__ = '0.2.2'
if sys.version_info[0] < 3:
    raise Exception("This only works with Python 3.5 or higher")
    if sys.version_info[1] < 5:
        raise Exception("This only works with Python 3.5 or higher")
try:
    from pkg_resources import parse_version
    if parse_version(torch.__version__) < parse_version("0.4.1"):
        raise Exception("PyTorch version should at least be 0.4.1")
except ImportError:
    # we silently ignore this and let the user run into problems later rather
    # than insist that setuptools must be installed
    pass

from gatelfpytorchjson.modelwrapperdefault import ModelWrapperDefault
from gatelfpytorchjson.modelwrapper import ModelWrapper
from gatelfpytorchjson.embeddingsmodule import EmbeddingsModule
from gatelfpytorchjson.takefromtuple import TakeFromTuple
from gatelfpytorchjson.CustomModule import CustomModule
from gatelfpytorchjson.ngrammodule import  NgramModule

