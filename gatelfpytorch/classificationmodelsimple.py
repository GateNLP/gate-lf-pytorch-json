from . modelbuilder import ModelBuilder

class ClassificationModelSimple(ModelBuilder):

    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset

    def train(self):
        pass