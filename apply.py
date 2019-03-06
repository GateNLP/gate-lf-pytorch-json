import sys
import json
import logging
import argparse
from pathlib import Path
import os
# make sure we can import the gatelfdata library from the default location
gatelfdatapath = os.path.join("..", "gate-lf-python-data")
filepath = os.path.dirname(__file__)
if filepath:
    gatelfdatapath = os.path.join(filepath, gatelfdatapath)
sys.path.append(gatelfdatapath)

from gatelfpytorchjson import utils
from gatelfpytorchjson import ModelWrapperDefault

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
    parser.add_argument("--labeled", action="store_true", help="Pass labeled instances instead just the feature vectors")
    parser.add_argument("--noret", action="store_true", help="Do not print the return value, only useful with labeled")
    parser.add_argument("modelname", help="Prefix of the model files pathnames (REQUIRED)")

    args = parser.parse_args(args=sysargs[1:])
    # not needed for now
    # metafile = args.metafile
    modelprefix = args.modelname

    # If we need the datadir
    # datadir = str(Path(modelprefix).parent)

    wrapper = ModelWrapperDefault.load(modelprefix, cuda=args.cuda, metafile=args.metafile)
    logger.info("DEBUG: model loaded:\n{}".format(wrapper.module))
    # get the target vocab
    vocab_target = wrapper.dataset.vocabs.get_vocab("<<TARGET>>")
    labels = vocab_target.itos

    ntotal = 0
    ncorrect = 0
    with sys.stdin as infile:
        for line in infile:
            logger.debug("Application input=%s" % (line,))
            if line == "STOP":
                break
            # NOTE: currently the LF always sends instances individually.
            # This may change in the future, but for now this simplifies dealing with the
            # special case where the LF can use the assigned class of the previous instance as a feature.
            # However we need to always apply to a set of instances, so wrap into another array here
            instancedata = json.loads(line)
            target  = None
            if args.labeled:
               target = instancedata[1]
               instancedata = instancedata[0]

            # NOTE: the  LF expects to get a map with the following elements:
            # status: must be "ok", anything else is interpreted as an error
            # output: the actual prediction: gets extracted from the returned data here
            # confidence: some confidence/probability score for the output, may be null: this gets extracted
            # from our returned data here
            # confidences: a map with confidences for all labels, may be null: this is NOT SUPPORTED in the LF yet!
            try:
                # NOTE: we put this into an extra list because the apply method expects a batch,
                # not a single instance
                # NOTE: the apply method also returns result for a whole batch!
                # print("DEBUG: calling wrapper.apply with instancedata=", instancedata, file=sys.stderr)
                batchof_labels, batchof_probs, batchof_probdists = wrapper.apply([instancedata])
                # NOTE: batchof_labels contains values for classification, but lists for
                # sequence tagging, so we check this first
                if not isinstance(batchof_labels, list):
                    raise Exception("Expected a list of predictions from apply but got %s" % (type(batchof_labels)))
                if len(batchof_labels) != 1:
                    raise Exception("Expected a list of length 1 (batchsize) but got %s" % (len(batchof_labels)))
                if isinstance(batchof_labels[0], list):
                    # we have a sequence tagging result
                    is_sequence = True
                else:
                    # we have a classification result
                    is_sequence = False
                output = batchof_labels[0]
                # print("DEBUG: output is", output, file=sys.stderr)
                if isinstance(batchof_probs, list) and len(batchof_probs) == 1:
                    prob = batchof_probs[0]
                    # NOTE: we still need to change the LF to handle this correctly!!!
                    # for now, just return prob as dist and prob[0] as prob/conf
                else:
                    prob = None
                if isinstance(batchof_probdists, list) and len(batchof_probdists) == 1:
                    dist = batchof_probdists[0]
                else:
                    dist = None
                ret = {"status": "ok", "output": output, "labels": labels, "conf": prob, "dist": dist}
            except Exception as e:
                logging.exception("Exception during processing of application result")
                ret = {"status": "error", "error": str(e)}
            logger.debug("Application result=%s" % (ret,))
            logger.debug("Ret=%r" % (ret,))
            retjson = json.dumps(ret)
            # print("DEBUG: returned JSON=", retjson, file=sys.stderr)
            if not args.noret:
                print(retjson)
                sys.stdout.flush()
            # now if we got labeled data, check if our stuff is correct, but only for single targets now
            if args.labeled and isinstance(target, str):
                if target == output:
                   ncorrect += 1
                ntotal += 1
    logger.debug("Application program terminating")
    if ntotal > 0:
        print("Total {}, correct {}, acc={}".format(ntotal, ncorrect, ncorrect/ntotal), file=sys.stderr)


if __name__ == '__main__':
    main(sys.argv)
