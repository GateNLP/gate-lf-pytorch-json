import torch.nn
from torch.autograd import Variable as V
import torch.nn.functional as F

class ClassificationModelSimple(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self):
        pass