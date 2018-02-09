
# Basic usage:
# ds = Dataset(metafile)
# model = MyModelBuilder(ds)
# model.train()
# # get some data for application some where
# instances = get_them()
# preditions = model.apply(instances)
# NOTE: maybe use the same naming conventions as scikit learn here!!

class ModelBuilder(object):

    # This requires an initialized dataset instance
    def __init__(self, dataset):
        pass

    # the implementation should figure out best values if parameter
    # is set to None
    # Also, by default, the method should decide which format
    # to use for reading the data (original or converted)
    def train(self, epochs=None, batchsize=None):
        pass