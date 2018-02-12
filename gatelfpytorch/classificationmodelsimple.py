import torch.nn
from torch.autograd import Variable as V
import torch.nn.functional as F
import sys

class ClassificationModelSimple(torch.nn.Module):

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

    def forward(self, batch):
        i_nom = 0
        i_ngr = 0
        input_layer_outputs = []
        for inputlayer, config in self.inputlayersinfo:
            inputtype = config["type"]
            if inputtype == "numeric":
                vals = [batch[i] for i in self.num_idxs]
                vals4pt = V(torch.FloatTensor(vals).t(), requires_grad=True)
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
        # concatenate the outputs
        hidden_vals = torch.cat(input_layer_outputs, 1)
        for hiddenlayer, config in self.hiddenlayersinfo:
            hidden_vals = hiddenlayer(hidden_vals)
        outputlayer, outputlayerconfig = self.outputlayerinfo
        out = outputlayer(hidden_vals)
        return out
