from . modelwrapper import ModelWrapper

class ModelWrapperSimple(ModelWrapper):

    # This requires an initialized dataset instance
    def __init__(self, dataset):
        super().__init__(dataset)


    # the implementation should figure out best values if parameter
    # is set to None
    # Also, by default, the method should decide which format
    # to use for reading the data (original or converted)
    def train(self, epochs=None, batchsize=None):
        pass