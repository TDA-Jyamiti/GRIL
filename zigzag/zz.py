from torch.utils.cpp_extension import load
import os.path as pth
from typing import List, Tuple
from torch import Tensor
from glob import glob
from torch.utils.cpp_extension import load
import torch
import mpml

__module_file_dir = pth.dirname(pth.realpath(__file__))
__cpp_src_dir = pth.join(__module_file_dir, '.')
src_files = []
extension = '*.cpp'
src_files.append(pth.join(__cpp_src_dir, 'multipers.cpp'))
__cuda_src_dir = pth.join(__module_file_dir, 'torchph/chofer_torchex/pershom/pershom_cpp_src')
__cuda_src_files = glob(__cuda_src_dir + '/*.cu')
# for cu_file in __cuda_src_files:
#     src_files.append(cu_file)
# extra_include_paths = pth.join(__cpp_src_dir, 'phat')
# src_files += [pth.join(__cpp_src_dir, 'phat')]
# jit compiling the c++ extension

# try:
#     __C = load(
#         'zigzag',
#         sources=src_files,
#         extra_cflags=['-fopenmp', '-O0'],
#         verbose=True)

# except Exception as ex:
#     print("Error was {}".format(ex))





class MultiPers:
    def __init__(self, hom_rank: int, l: int, res: float, ranks: List[int]):
        # try:
        #     __M = load(
        #         'zigzag',
        #         sources=src_files,
        #         extra_cflags=['-fopenmp', '-O0'],
        #         verbose=True)

        # except Exception as ex:
        #     print("Error was {}".format(ex))
        
        self.mpl = mpml.Multipers(hom_rank, l, res, ranks)
    
    
    def compute_landscape(self, pts: List[Tuple[int]], batch: List[Tuple[Tensor, List[List[int]]]]):
        return self.mpl.compute_landscape(pts, batch)

    def set_max_jobs(self, njobs: int):
        self.mpl.set_max_jobs(njobs)
        

"""


def zigzag_pairs(num_vertices, simplices_birth_death: List[Tuple], boundary_map: Tensor, manual_birth_pts: int,
                 manual_death_pts: int):
    return __C.zigzag_pairs(num_vertices, simplices_birth_death, boundary_map, manual_birth_pts, manual_death_pts)


# INPUT: (f, e, num_vertices, [p1, p2, p3, ..., pm])

def compute_landscape(pts: List[Tuple], hom_rank: int, reqd_rank_1: int, reqd_rank_2: int, pers_inp: List[Tuple[Tensor, List[List[int]], int]]):
    return __C.compute_landscape_batch(pts, hom_rank, reqd_rank_1, reqd_rank_2, pers_inp)

def compute_landscape_multi_rank(pts: List[Tuple], hom_rank: int, reqd_ranks: List[int], pers_inp: List[Tuple[Tensor, List[List[int]], int]]):
    return __C.compute_landscape_batch_multi_rank(pts, hom_rank, reqd_ranks, pers_inp)
"""