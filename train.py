import sys
import os
import logging
from gatelfdata import Dataset
from gatelfpytorchjson import ModelWrapperSimple
from gatelfpytorchjson import ModelWrapper
import argparse
from pathlib import Path

metafile = sys.argv[1]
modelname = sys.argv[2]
datadir = Path(metafile).parent

parms = sys.argv[3:]

# The way how argparse treats boolean arguments sucks, so we need to do this
def str2bool(val):
    if val.lower() in ["yes", "true", "y", "t", "1"]:
        return True
    elif val.lower() in ["no", "false", "n", "f", "0"]:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected, not %s" % (val,))

parser = argparse.ArgumentParser()
# Parameter for checkpointing the module
parser.add_argument("--embs", type=str, help="Override embedding settings, specify as embid:embdims:embtrain:embminfreq,embid:embdims ..")
parser.add_argument("--valsize", type=float, help="Set the validation set size (>1) or proportion (<1)")
parser.add_argument("--valevery", type=int, default=10, help="Evaluate on validation set and log every that many batches")
parser.add_argument("--batchsize", type=int, default=32, help="Batch size")
parser.add_argument("--maxepochs", type=int, default=50, help="Maximum number of epochs")
parser.add_argument("--stopfile", type=str, help="If that file exists, training is stopped")
parser.add_argument("--module", type=str, help="The class/file name to use for the pytorch module")
parser.add_argument("--learningrate", type=float, help="Override default learning rate for the optimizer")
parser.add_argument("--cuda", type=str2bool, help="True/False to use CUDA or not, omit to determine automatically")
# NOTE: resume currently does not make sure that the original metafile info is used (but maybe new data):
# This should work once the metadata is actually stored as part of the model!
parser.add_argument("--resume", action='store_true', help="Resume training from the specified model")
parser.add_argument("--notrain", action='store_true', help="Do not actually run training (for use with LF)")

args = parser.parse_args(parms)
config = vars(args)

# Set up logging
logger1 = logging.getLogger("gatelfdata")
logger1.setLevel(logging.INFO)
logger2 = logging.getLogger("gatelfpytorchjson")
logger2.setLevel(logging.DEBUG)
logger3 = logging.getLogger(__name__)
logger3.setLevel(logging.DEBUG)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
filehandler = logging.FileHandler(os.path.join(datadir, "pytorch-json.train.log"))
filehandler.setFormatter(formatter)
logger1.addHandler(filehandler)
logger1.addHandler(streamhandler)
logger2.addHandler(filehandler)
logger2.addHandler(streamhandler)
logger3.addHandler(filehandler)
logger3.addHandler(streamhandler)


# TODO: use a static Dataset method to parse the remaining args and create an args
# dict to pass to the constructors of Dataset and the wrapper so each can pick the 
# parameters relevant to them!

logger3.debug("Running train.py, config is %r" % config)
if config.get("notrain"):
    logger3.info("--notrain specified, exiting")
    sys.exit(0)

logger3.debug("Loading metafile...")
ds = Dataset(metafile, config=config)
logger3.debug("Metafile loaded.")

# TODO: test passing on parameters
if config.get("resume"):
    logger3.info("--resume specified, loading and continuing on existing model")
    wrapper = ModelWrapperSimple.load(modelname)
    logger3.debug("Modelwrapper loaded")
    logger3.debug("Model is %r" % wrapper)
else:
    logger3.debug("Creating ModelWrapperSimple")
    wrapper = ModelWrapperSimple(ds, config=config)
    logger3.debug("Modelwrapper created")
    logger3.debug("Model is %r" % wrapper)

# TODO: this may need to be done differently if we have our own validation file!
# TODO: the default to use for validation set size should be settable through config in the constructor!
logger3.debug("Preparing the data...")
valsize = config.get("valsize")
if valsize:
    wrapper.prepare_data(validationsize=valsize)
else:
    wrapper.prepare_data()
logger3.debug("Data prepared")

# TODO: check if setters or using constructor parameters make more sense here
wrapper.validate_every_batches = config["valevery"]

# TODO: figure out what good defaults are here and what we want to set here rather than
# in the constructor. Maybe allow to set everything in the constructor for simplicity?
logger3.debug("Start training...")
wrapper.train(batch_size=config["batchsize"], early_stopping=False, max_epochs=config["maxepochs"])
logger3.debug("Training completed")

logger3.debug("Saving model...")
wrapper.save(modelname)
logger3.debug("Model saved")

# print the model used again so we do not have to scoll back a huge log ...
logger3.info("Model: %r" % wrapper)
