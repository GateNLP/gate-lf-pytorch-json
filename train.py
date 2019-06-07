import sys
import os
import logging
from configsimple import topconfig

# make sure we can import the gatelfdata library from the default location
gatelfdatapath = os.path.join("..", "gate-lf-python-data")
filepath = os.path.dirname(__file__)
if filepath:
    gatelfdatapath = os.path.join(filepath, gatelfdatapath)
sys.path.append(gatelfdatapath)
from gatelfdata import Dataset
from gatelfpytorchjson import ModelWrapper
from gatelfpytorchjson import ModelWrapperDefault
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
    myconfig = ModelWrapperDefault.configsimple(topconfig)

    myconfig.parse_args(args=sysargs[1:])

    if myconfig.get("es_metric") not in ["loss", "accuracy"]:
        raise Exception("es_metric must be loss or accuracy")

    metafile = myconfig.get("metafile")
    modelname = myconfig.get("modelname")

    if not metafile or not modelname:
        raise Exception("Metafile or modelfile not specified, use --help parameter for help\nLearningFramework defaults are: crvd.meta.json FileJsonPyTorch.model")

    datadir = str(Path(metafile).parent)

    if myconfig.get("debug"):
        logger.setLevel(logging.DEBUG)

    es_patience = myconfig.get("es_patience")
    es_mindelta = 0.0

    def es_lambda(losses=None, accs=None, patience=None, mindelta=None, metric="loss"):
        return ModelWrapper.\
            early_stopping_checker(losses, accs, patience=es_patience, mindelta=es_mindelta,
                                   metric=myconfig["es_metric"])

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

    logger.debug("Running train.py, config is %r" % myconfig)

    if myconfig.get("nocreate"):
        logger.info("--nocreate specified, exiting")
        sys.exit(0)

    logger.debug("Loading metafile...")
    ds = Dataset(metafile, config=myconfig)
    logger.debug("Metafile loaded.")

    # determine and use the correct modelwrapper
    # default is ModelWrapperSimple
    wrapper_class = ModelWrapperDefault
    if myconfig.get("wrapper"):
        wrapperclassname = myconfig["wrapper"]
        import importlib
        module = importlib.import_module("gatelfpytorchjson." + wrapperclassname)
        wrapper_class_ = getattr(module, wrapperclassname)

    # TODO: test passing on parameters
    if myconfig.get("resume"):
        logger.info("--resume specified, loading and continuing on existing model")
        wrapper = wrapper_class.load(modelname, metafile=metafile)
        logger.debug("Modelwrapper loaded")
        logger.debug("Model is %r" % wrapper)
    else:
        logger.debug("Creating ModelWrapperSimple")
        wrapper = wrapper_class(ds, config=myconfig)
        logger.debug("Modelwrapper created")
        logger.debug("Model is %r" % wrapper)

    if myconfig.get("notrain"):
        logger.info("--notrain specified, exiting")
        sys.exit(0)

    if myconfig.get("debug"):
        glf = getattr(wrapper, "get_logger", None)
        if glf and callable(glf):
            wlogger = wrapper.get_logger()
            logger.debug("Setting wrapper logging level to DEBUG")
            wlogger.setLevel(logging.DEBUG)
        else:
            logger.debug("Wrapper has not logging, cannot set to DEBUG")

    # TODO: the default to use for validation set size should be settable through config in the constructor!
    logger.debug("Preparing the data...")
    # if we have a validation file, use it, ignore the valsize
    if myconfig.get("valfile"):
        wrapper.prepare_data(file=myconfig["valfile"])
    else:
        valsize = myconfig.get("valsize")
        if valsize is not None:
            wrapper.prepare_data(validationsize=valsize)
        else:
            wrapper.prepare_data()
    logger.debug("Data prepared")

    wrapper.validate_every_batches = myconfig["valeverybatches"]
    wrapper.validate_every_epochs = myconfig["valeveryepochs"]
    wrapper.validate_every_instances = myconfig["valeveryinstances"]
    wrapper.report_every_instances = myconfig["repeveryinstances"]
    wrapper.report_every_batches = myconfig["repeverybatches"]

    # TODO: figure out what good defaults are here and what we want to set here rather than
    # in the constructor. Maybe allow to set everything in the constructor for simplicity?
    logger.info("Model: %r" % wrapper)
    logger.debug("Start training...")
    wrapper.train(batch_size=myconfig["batchsize"],
                  early_stopping=es_lambda, max_epochs=myconfig["maxepochs"], filenameprefix=modelname
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
