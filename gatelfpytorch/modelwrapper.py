
# Basic usage:
# ds = Dataset(metafile)
# wrapper = ModelWrapperSimple(ds) # or some other subclass
# wrapper.train()
# # get some data for application some where
# instances = get_them()
# preditions = wrapper.apply(instances)
# NOTE: maybe use the same naming conventions as scikit learn here!!

class ModelWrapper(object):

    # This requires an initialized dataset instance
    def __init__(self, dataset):
        self.num_idxs = dataset.get_numeric_feature_idxs()
        self.nom.idxs = dataset.get_nominal_feature_idxs()
        self.ngr.idxs = dataset.get_ngram_feature_idxs()
        num_feats = dataset.get_numeric_features()
        nom_feats = dataset.get_nominal_features()
        ngr_feats = dataset.get_ngram_features()


    # the implementation should figure out best values if parameter
    # is set to None
    # Also, by default, the method should decide which format
    # to use for reading the data (original or converted)
    def train(self, epochs=None, batchsize=None):
        # for each input, pass it the corresponding feature or features
        # then concatenate all the outputs to form the first hidden layer
