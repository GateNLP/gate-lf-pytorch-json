"""Various commonly used utility functions"""

import argparse


# The way how argparse treats boolean arguments sucks, so we need to do this
def str2bool(val):
    val = str(val)
    if val.lower() in ["yes", "true", "y", "t", "1"]:
        return True
    elif val.lower() in ["no", "false", "n", "f", "0"]:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected, not %s" % (val,))
