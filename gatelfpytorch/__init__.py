# check minimum requirements: python 3 and minimum torch version
import sys
import torch
if sys.version_info[0] < 3:
    raise Exception("This only works with Python 3 or higher")
try:
    from pkg_resources import parse_version
    if parse_version(torch.__version__) < parse_version("0.3.0"):
        raise Exception("PyTorch version should at least be 0.3.0")
except ImportError:
    # we silently ignore this and let the user run into problems later rather
    # than insist that setuptools must be installed
    pass

from .modelwrappersimple import ModelWrapperSimple
