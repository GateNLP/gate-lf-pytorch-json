import torch.nn
from torch.autograd import Variable as V
import torch.nn.functional as F


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
                vals4pt = self.convert_nominal(batch[nom_idx])
                out = inputlayer(vals4pt)
                input_layer_outputs.append(out)
            elif inputtype == "ngram":
                ngr_idx = self.ngr_idxs[i_ngr]
                i_ngr += 1
                vals4pt = self.convert_ngram(batch[ngr_idx])
                out = inputlayer(vals4pt)
                input_layer_outputs.append(out)
            else:
                raise Exception("Odd input type: %s" % inputtype)
            # concatenate the outputs
            hidden_vals = torch.cat(input_layer_outputs)
            for hiddenlayer, config in self.hiddenlayersinfo:
                hidden_vals = hiddenlayer(hidden_vals)
            outputlayer, outputlayerconfig = self.outputlayerinfo
            out = outputlayer(hidden_vals)
            return out
