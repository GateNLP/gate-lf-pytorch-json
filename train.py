import sys
import os
import logging
from gatelfdata import Dataset
from gatelfpytorchjson import ModelWrapperSimple
from gatelfpytorchjson import ModelWrapper
import argparse

metafile=sys.argv[1]
modelname=sys.argv[2]

parms=sys.argv[3:]

parser = argparse.ArgumentParser()
parser.add_argument("--embs", type=str, help="Override embedding settings, specify as embid:embdims:embtrain:embminfreq,embid:embdims ..")
parser.add_argument("--valsize", type=float, help="Set the validation set size (>1) or proportion (<1)")
parser.add_argument("--valevery", type=int, default=10, help="Validate and log every that many batches")
parser.add_argument("--batchsize", type=int, default=32, help="Batch size")
parser.add_argument("--maxepochs", type=int, default=50, help="Maximum number of epochs")
parser.add_argument("--stopfile", type=str, help="If that file exists, training is stopped")
parser.add_argument("--learningrate", type=float, help="Override default learning rate for the optimizer")
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
filehandler = logging.FileHandler("training.log")
logger1.addHandler(filehandler)
logger1.addHandler(streamhandler)
logger2.addHandler(filehandler)
logger2.addHandler(streamhandler)
logger3.addHandler(filehandler)
logger3.addHandler(streamhandler)


# TODO: use a static Dataset method to parse the remaining args and create an args
# dict to pass to the constructors of Dataset and the wrapper so each can pick the 
# parameters relevant to them!

logger3.debug("Loading metafile...")
ds = Dataset(metafile, config=config)
logger3.debug("Metafile loaded.")

# TODO: test passing on parameters
logger3.debug("Creating ModelWrapperSimple")
wrapper = ModelWrapperSimple(ds, cuda=False, config=config)
logger3.debug("Modelwrapper created")
logger3.debug("Model is %r" % wrapper)

# TODO: this may need to be done differently if we have our own validation file!
# TODO: the default to use for validation set size should be settable through config in the constructor!
logger3.debug("Preparing the data...")
valsize=config.get("valsize")
if valsize:
    wrapper.prepare_data(validationsize=200)
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
