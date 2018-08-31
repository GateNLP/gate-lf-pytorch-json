import sys
import json
import logging
from gatelfpytorchjson import ModelWrapperDefault
import argparse
from pathlib import Path
from gatelfpytorchjson import utils
import os
# make sure we can import the gatelfdata library from the default location
gatelfdatapath = os.path.join("..", "gate-lf-python-data")
filepath = os.path.dirname(__file__)
if filepath:
    gatelfdatapath = os.path.join(filepath, gatelfdatapath)
sys.path.append(gatelfdatapath)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)


def main(sysargs):

    logger.debug("PYTHON APPLICATION SCRIPT, args=%s" % (sys.argv,))

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=utils.str2bool, default=False, help="True/False to use CUDA or not, default is False")
    parser.add_argument("--metafile", type=str, default=None, help="Meta file, if necessary")
    parser.add_argument("modelname", help="Prefix of the model files pathnames (REQUIRED)")

    args = parser.parse_args(args=sysargs[1:])
    # not needed for now
    # metafile = args.metafile
    modelprefix = args.modelname

    # If we need the datadir
    # datadir = str(Path(modelprefix).parent)

    wrapper = ModelWrapperDefault.load(modelprefix)
    # TODO: set cuda depending on the parameter, if necessary
    # ??? Should we actually ever use the GPU for application?
    wrapper.set_cuda(args.cuda)

    with sys.stdin as infile:
        for line in infile:
            print("PYTHON APPLICATION, input=", line, file=sys.stderr)
            if line == "STOP":
                break
            # TODO: currently the LF sends individual instances here, we may want to change
            # However we need to always apply to a set of instances, so wrap into another array
            instancedata = json.loads(line)
            # TODO: better error handling: put the apply call into a try block and catch any error, also
            # check returned data. If there is a problem send back in the map we return!!
            # NOTE: the  LF expects to get a map with the following elements:
            # status: must be "ok", anything else is interpreted as an error
            # output: the actual prediction: gets extracted from the returned data here
            # confidence: some confidence/probability score for the output, may be null: this gets extracted
            # from our returned data here
            # confidences: a map with confidences for all labels, may be null: this is NOT SUPPORTED in the LF yet!
            preds=wrapper.apply([instancedata])
            print("PYTHON APPLICATION, preds=", preds, file=sys.stderr)
            # preds are a list of one or two lists, where the first list contains all the labels and the second
            # list contains all the confidences in the order used by the model.
            # For now we just extract the label or for a sequence, the list of labels,
            # knowing that for now we always process only one instance/sequence!
            ret = {"status":"ok", "output":preds[0][0]}
            print("PYTHON APPLICATION, return=", ret, file=sys.stderr)
            print(json.dumps(ret))
            # TODO: IMPORTANT!!! What the model returns is currently different from what the LF code expects!!!
            sys.stdout.flush()
    print("PYTHON APPLICATION SCRIPT: finishing",file=sys.stderr)


if __name__ == '__main__':
    main(sys.argv)