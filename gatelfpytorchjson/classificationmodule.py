import torch.nn
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)


class ClassificationModule(torch.nn.Module):

    def __init__(self, inputlayersinfo, hiddenlayersinfo, outputlayerinfo,
                 featureconfig):
        """Construct the model: inputlayers is a list of tuples, where
        the first element is the PyTorch module and the second element
        is a dictionary of config data for that layer. The config data
        must contain the "type" key. Hiddenlayers is a  similar list
        of hidden layers, where the first element of the list corresponds
        to the layer after the input layer and the last element to
        the layer before the output layer.
        Output layer is a single tuple (module,configs).
        The featureconfig parameter contains the feature indices
        per type and maybe other config info created by the wrapper.
        """
        super().__init__()
        self.inputlayersinfo = inputlayersinfo
        self.hiddenlayersinfo = hiddenlayersinfo
        self.outputlayerinfo = outputlayerinfo
        self.num_idxs = featureconfig["num_idxs"]
        self.nom_idxs = featureconfig["nom_idxs"]
        self.ngr_idxs = featureconfig["ngr_idxs"]
        # register all the layers with the module so PyTorch knows
        # about parameters etc.
        for layer, config in inputlayersinfo:
            self.add_module(config.get("name"), layer)
        for layer, config in hiddenlayersinfo:
            self.add_module(config.get("name"), layer)
        outlayer = outputlayerinfo[0]
        outconfig = outputlayerinfo[1]
        self.add_module(outconfig.get("name"), outlayer)
        self._on_cuda = None

    def set_seed(self, seed):
        torch.manual_seed(seed)
        # make sure it is set on all GPUs as well, we can always do this as torch ignores
        # this if no CUDA is available
        torch.cuda.manual_seed_all(seed)

    def on_cuda(self):
        """Returns true or false depending on if the module is on cuda or not. Unfortunately
        there is no API method in PyTorch for this so we get this from the first parameter of the
        model and cache it."""
        if self._on_cuda is None:
            self._on_cuda = next(self.parameters()).is_cuda
        return self._on_cuda

    def forward(self, batch):
        # logger.debug("Calling forward with %s" % (batch,))
        # logger.debug("inputlayersinfo is %s" % (self.inputlayersinfo,))
        i_nom = 0
        i_ngr = 0
        input_layer_outputs = []
        # TODO: when we use the Concat layer instead of manually concatenating,
        # we just create all the input variables her and append to the inputs list,
        # then pass this on to the concat layer.
        for inputlayer, config in self.inputlayersinfo:
            inputtype = config["type"]
            if inputtype == "numeric":
                vals = [batch[i] for i in self.num_idxs]
                vals4pt = torch.FloatTensor(vals).t()
                vals4pt.requires_grad_(True)
                if self.on_cuda():
                    vals4pt = vals4pt.cuda()
                out = inputlayer(vals4pt)
                input_layer_outputs.append(out)
            elif inputtype == "nominal":
                nom_idx = self.nom_idxs[i_nom]
                i_nom += 1
                # the EmbeddingsModule takes the original converted batch values, not a Tensor or Variable
                # vals4pt = V(torch.LongTensor(batch[nom_idx]), requires_grad=False)
                out = inputlayer(batch[nom_idx])
                input_layer_outputs.append(out)
            elif inputtype == "ngram":
                ngr_idx = self.ngr_idxs[i_ngr]
                i_ngr += 1
                out = inputlayer(batch[ngr_idx])
                input_layer_outputs.append(out)
            else:
                raise Exception("Odd input type: %s" % inputtype)
        # concatenate the outputs, i.e. the last dimension
        hidden_vals = torch.cat(input_layer_outputs, len(input_layer_outputs[0].size())-1)
        for hiddenlayer, config in self.hiddenlayersinfo:
            # print("DEBUG: Have shape: ", hidden_vals.size(),  file=sys.stderr)
            # print("DEBUG: Trying to apply hidden layer: ", hiddenlayer,  file=sys.stderr)
            # TODO: IMPORTANT: if we have an LSTM somewhere, the lstm returns a tuple, so passing
            # it on to the next layer will not work!! Instead we need to wrap the LSTM into a
            # takefromtuple layer.
            hidden_vals = hiddenlayer(hidden_vals)
        outputlayer, outputlayerconfig = self.outputlayerinfo
        out = outputlayer(hidden_vals)
        return out

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_on_cuda"]
        return state

    def __setstate__(self, state):
        state["_on_cuda"] = None
        self.__dict__.update(state)
