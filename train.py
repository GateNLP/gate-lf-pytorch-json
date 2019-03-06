import sys
import os
import logging
# make sure we can import the gatelfdata library from the default location
gatelfdatapath = os.path.join("..", "gate-lf-python-data")
filepath = os.path.dirname(__file__)
if filepath:
    gatelfdatapath = os.path.join(filepath, gatelfdatapath)
sys.path.append(gatelfdatapath)
import gatelfdata
import gatelfpytorchjson
from gatelfdata import Dataset
from gatelfpytorchjson import ModelWrapper
from gatelfpytorchjson import ModelWrapperDefault
from gatelfpytorchjson import utils
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)


def main(sysargs):

    logger.debug("Called with args=%s" % (sysargs,))
    parser = argparse.ArgumentParser()
    parser.add_argument("--embs", type=str, help="Override embedding settings, specify as embid:embdims:embtrain:embminfreq:embfile,embid:embdims ..")
    parser.add_argument("--valsize", type=float, help="Set the validation set size (>1) or proportion (<1)")
    parser.add_argument("--valeverybatches", type=int, default=None, help="Evaluate on validation set and log every that many batches (None)")
    parser.add_argument("--valeveryepochs", type=int, default=1, help="Evaluate on validation set and log every that many epochs (1)")
    parser.add_argument("--valeveryinstances", type=int, default=None, help="Evaluate on validation set and log every that many instances (None)")
    parser.add_argument("--repeveryinstances", type=int, default=500, help="Report on training set and log every that many instances (500)")
    parser.add_argument("--repeverybatches", type=int, default=None, help="Report on training set and log every that many batches (None)")
    parser.add_argument("--batchsize", type=int, default=32, help="Batch size")
    parser.add_argument("--maxepochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--stopfile", type=str, help="If that file exists, training is stopped")
    parser.add_argument("--module", type=str, help="The class/file name to use for the pytorch module (within modelzoo)")
    parser.add_argument("--wrapper", type=str, help="The class/file name to use as the model wrapper")
    parser.add_argument("--learningrate", type=float, help="Override default learning rate for the optimizer")
    parser.add_argument("--ngram_layer", type=str, default="cnn", help="Architecture to use for ngrams: lstm or cnn (cnn)")
    parser.add_argument("--es_patience", type=int, default=2, help="Early stopping patience iterations (2)")
    parser.add_argument("--es_metric", type=str, default="loss", help="Which metric to use for early stopping, 'loss' or 'accuracy' (loss)")
    parser.add_argument("--cuda", type=utils.str2bool, help="True/False to use CUDA or not, omit to determine automatically")
    parser.add_argument("--seed", type=int, default=0, help="Random seed to make experiments repeatable/explore randomness (default 0=random random seed)")
    # NOTE: resume currently does not make sure that the original metafile info is used (but maybe new data):
    # This should work once the metadata is actually stored as part of the model!
    parser.add_argument("--resume", action='store_true', help="Resume training from the specified model")
    parser.add_argument("--notrain", action='store_true', help="Do not actually run training, but show generated model")
    parser.add_argument("--nocreate", action='store_true', help="Do not actually even create module (do nothing)")
    parser.add_argument("--valfile", type=str, default=None, help="Use this file for validation")
    parser.add_argument("--version", action='version', version=gatelfpytorchjson.__version__)
    parser.add_argument("--debug", action='store_true', help="Set logger to DEBUG and show more information")
    parser.add_argument("metafile", help="Path to metafile (REQUIRED)")
    parser.add_argument("modelname", help="Model path prefix (full path and beginning of model file name) (REQUIRED)")

    args = parser.parse_args(args=sysargs[1:])

    if args.es_metric not in ["loss", "accuracy"]:
        raise Exception("es_metric must be loss or accuracy")

    metafile = args.metafile
    modelname = args.modelname

    if not metafile or not modelname:
        raise Exception("Metafile or modelfile not specified, use --help parameter for help\nLearningFramework defaults are: crvd.meta.json FileJsonPyTorch.model")

    datadir = str(Path(metafile).parent)

    config = vars(args)

    if config.get("debug"):
        logger.setLevel(logging.DEBUG)

    es_patience = config.get("es_patience")
    es_mindelta = 0.0
    def es_lambda(losses=None, accs=None, patience=None, mindelta=None, metric="loss"):
        return ModelWrapper.\
            early_stopping_checker(losses, accs, patience=es_patience, mindelta=es_mindelta, metric=config["es_metric"])


    # Also log to a file
    filehandler = logging.FileHandler(os.path.join(datadir, "pytorch-json.train.log"))
    filehandler.setFormatter(formatter)

    # in order to override the logging level of any of the modules/classes used,
    # get the logger and do it here
    # logger1 = logging.getLogger("gatelfpytorchjson.modelwrapperdefault")
    # logger1.setLevel(logging.DEBUG)


    # TODO: use a static Dataset method to parse the remaining args and create an args
    # dict to pass to the constructors of Dataset and the wrapper so each can pick the
    # parameters relevant to them!

    logger.debug("Running train.py, config is %r" % config)

    if config.get("nocreate"):
        logger.info("--nocreate specified, exiting")
        sys.exit(0)


    logger.debug("Loading metafile...")
    ds = Dataset(metafile, config=config)
    logger.debug("Metafile loaded.")

    # determine and use the correct modelwrapper
    # default is ModelWrapperSimple
    wrapper_class = ModelWrapperDefault
    if config.get("wrapper"):
        wrapperclassname = config["wrapper"]
        print("!!!!!DEBUG: trying to use wrapper class/file: ", wrapperclassname, file=sys.stderr)
        import importlib
        module = importlib.import_module("gatelfpytorchjson." + wrapperclassname)
        wrapper_class_ = getattr(module, wrapperclassname)

    # TODO: test passing on parameters
    if config.get("resume"):
        logger.info("--resume specified, loading and continuing on existing model")
        wrapper = wrapper_class.load(modelname, metafile=metafile)
        logger.debug("Modelwrapper loaded")
        logger.debug("Model is %r" % wrapper)
    else:
        logger.debug("Creating ModelWrapperSimple")
        wrapper = wrapper_class(ds, config=config)
        logger.debug("Modelwrapper created")
        logger.debug("Model is %r" % wrapper)

    if config.get("notrain"):
        logger.info("--notrain specified, exiting")
        sys.exit(0)

    if config.get("debug"):
        glf = getattr(wrapper, "get_logger", None)
        if  glf and callable(glf):
            wlogger = wrapper.get_logger()
            logger.debug("Setting wrapper logging level to DEBUG")
            wlogger.setLevel(logging.DEBUG)
        else:
            logger.debug("Wrapper has not logging, cannot set to DEBUG")

    # TODO: the default to use for validation set size should be settable through config in the constructor!
    logger.debug("Preparing the data...")
    # if we have a validation file, use it, ignore the valsize
    if config.get("valfile"):
        wrapper.prepare_data(file=config["valfile"])
    else:
        valsize = config.get("valsize")
        if valsize is not None:
            wrapper.prepare_data(validationsize=valsize)
        else:
            wrapper.prepare_data()
    logger.debug("Data prepared")

    wrapper.validate_every_batches = config["valeverybatches"]
    wrapper.validate_every_epochs = config["valeveryepochs"]
    wrapper.validate_every_instances = config["valeveryinstances"]
    wrapper.report_every_instances = config["repeveryinstances"]
    wrapper.report_every_batches = config["repeverybatches"]

    # TODO: figure out what good defaults are here and what we want to set here rather than
    # in the constructor. Maybe allow to set everything in the constructor for simplicity?
    logger.info("Model: %r" % wrapper)
    logger.debug("Start training...")
    wrapper.train(batch_size=config["batchsize"],
                  early_stopping=es_lambda, max_epochs=config["maxepochs"], filenameprefix=modelname
                  )
    logger.debug("Training completed")

    # NOTE: this will save the modelwrapper, and will ONLY save the model if we did not already
    # save the best model during training!
    logger.debug("Saving model...")
    wrapper.save(modelname)
    logger.debug("Model saved")

    # print the model used again so we do not have to scoll back a huge log ...
    logger.info("Model: %r" % wrapper)


if __name__ == '__main__':
    main(sys.argv)
