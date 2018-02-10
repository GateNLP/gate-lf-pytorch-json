
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
<<<<<<< HEAD:gatelfpytorch/modelwrapper.py

=======
>>>>>>> 099486cd59e4b143ee8a1591a032e291970c4c99:gatelfpytorch/modelbuilder.py
        pass

    # the implementation should figure out best values if parameter
    # is set to None
    # Also, by default, the method should decide which format
    # to use for reading the data (original or converted)
    def train(self, epochs=None, batchsize=None):
        pass