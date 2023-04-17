from torch.utils.cpp_extension import load
from torch import Tensor

test = load('test', sources=['test.cpp'], extra_cflags= ['-fopenmp'], verbose=True)

test.test(32)