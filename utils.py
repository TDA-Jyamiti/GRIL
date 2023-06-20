from random import sample
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from scipy import sparse
from typing import Tuple, List
import argparse
import gril.gril as gril
import torch
import torch.nn as nn
from scipy.spatial import Delaunay


def delaunay_complex(x):
    edge_dict = {}
    edges = []
    tri_converted = []
    tri = Delaunay(x)
    triangles = tri.simplices
    for t in triangles:
        s = sorted(t)
        tri_boundary = []
        for i in range(len(s)):
            edge_key = tuple(sorted(s[:i] + s[i + 1:]))
            if edge_key not in edge_dict:
                edge_dict[edge_key] = len(edge_dict)
                edges.append(list(edge_key))
            tri_boundary.append(edge_dict[edge_key])
        tri_converted.append(tri_boundary)
    edges_t = torch.tensor(edges, dtype=torch.long)
    tri_t = torch.tensor(triangles, dtype=torch.long)
    tri_converted_t = torch.tensor(tri_converted, dtype=torch.long)
    return edges_t, tri_t, tri_converted_t

def pre_process_edges(edge_index):
    e = edge_index.permute(1, 0)
    e = e.sort(1)
    e = e[0].tolist()
    e = set([tuple(ee) for ee in e])
    return torch.tensor([ee for ee in e], dtype=torch.long, device=edge_index.device)


def get_simplices(num_vertices, edges, triangles=None):
    simp = [[i] for i in range(num_vertices)]
    for e in edges:
        e_ = sorted([e[0].item(), e[1].item()])
        simp.append(e_)
    if triangles is not None:
        for f in triangles:
            f_ = sorted([f[0].item(), f[1].item(), f[2].item()])
            simp.append(f_)
    return simp

def get_filtration(x, edges, tri, tri_converted, nn_k=6):
    # edges = pre_process_edges(edge_index)
    d_xx = torch.cdist(x, x)
    nn_k = min([nn_k, x.shape[0]])
    d_xx = d_xx.topk(nn_k, 1, largest=False).values
    # d_xy = -(d_xx * d_xx)
    d_xx = -d_xx[:, 1:]
    d_xx = torch.exp(d_xx).sum(1) / nn_k 
    d_xx = 1 - d_xx
    # d_xx = d_xx * 3.;
    # d_xx = d_xx / d_xx.max()
    e_val = d_xx.unsqueeze(0).expand((edges.size(0), -1))
    e_val = e_val.gather(1, edges)
    e_val = e_val.max(1)[0]

    tri_val = d_xx.unsqueeze(0).expand((tri.size(0), -1))
    tri_val = tri_val.gather(1, tri)
    tri_val = tri_val.max(1)[0]

    e_val_x = x[edges[:, 0]]
    e_val_y = x[edges[:, 1]]
    e_val_y = torch.norm(e_val_x - e_val_y, dim=1)
    e_val_y = 1 - torch.exp(-e_val_y)

    tri_val_2 = e_val_y.unsqueeze(0).expand((tri_converted.size(0), -1))
    tri_val_2 = tri_val_2.gather(1, tri_converted)
    tri_val_2 = tri_val_2.max(1)[0]

    f_v = torch.cat([d_xx.view((-1, 1)), torch.zeros((d_xx.size(0), 1))], dim=1)
    e_val = torch.cat([e_val.view((-1, 1)), e_val_y.view((-1, 1))], dim=1) + 0.01
    tri_val = torch.cat([tri_val.view((-1, 1)), tri_val_2.view((-1, 1))], dim=1) + 0.02

    filt = torch.cat([f_v, e_val, tri_val], dim=0)
    # filt = torch.cat([f_v, e_val], dim=0)
    return filt, edges

def create_circle(center, radius, w_noise=False, num_points=100):
    np.random.seed(0)
    theta = np.random.uniform(size=(num_points,)) * np.pi * 2
    data_x = radius * np.cos(theta) + center[0]
    data_y = radius * np.sin(theta) + center[1]
    data = np.column_stack((data_x, data_y))
    if w_noise:
            np.random.seed(0)
            U1 = np.random.uniform(size=10)
            np.random.seed(42)
            U2 = np.random.uniform(size=10)
            # np.random.seed(0)
            # U3 = np.random.uniform(0, 1, (10, 2))
            data_noise_x = 0.1 * np.sqrt(U2) * np.cos(2 * np.pi * U1) + center[0]
            data_noise_y = 0.1 * np.sqrt(U2) * np.sin(2 * np.pi * U1) + center[1]
            data_noise = np.column_stack([data_noise_x, data_noise_y])
            data = np.row_stack([data, data_noise])
    return data

def create_disk(center, radius, num_points=100):
    np.random.seed(0)
    U1 = np.random.uniform(size=num_points)
    np.random.seed(42)
    U2 = np.random.uniform(size=num_points)
    data_x = radius * np.sqrt(U2) * np.cos(2 * np.pi * U1) + center[0]
    data_y = radius * np.sqrt(U2) * np.sin(2 * np.pi * U1) + center[1]
    data = np.column_stack((data_x, data_y))
    return data



def create_circles(n_circles=2, w_noise=False):
    circles = []
    # centers = [[0.15 + (i * 0.80), (0.35 + i * 0.15)] for i in range(n_circles)]
    centers = [[0.15, 0.2], [0.75, 0.9], [0.2, 0.7]]
    radius = [0.15, 0.17, 0.2]
    centers = np.array(centers)
    for i in range(n_circles):
        np.random.seed(0)
        theta = np.random.uniform(size=(100,)) * np.pi * 2
        data_x = radius[i] * np.cos(theta).reshape((theta.shape[0], 1)) + centers[i][0]
        data_y = radius[i] * np.sin(theta).reshape((theta.shape[0], 1)) + centers[i][1]
        data = np.concatenate([data_x, data_y], axis=1)
        if w_noise:
            np.random.seed(0)
            U1 = np.random.uniform(size=10)
            np.random.seed(42)
            U2 = np.random.uniform(size=10)
            r = 0.1
            data_noise_x = r * np.sqrt(U2) * np.cos(2 * np.pi * U1) + centers[i][0]
            data_noise_y = r * np.sqrt(U2) * np.sin(2 * np.pi * U1) + centers[i][1]
            data_noise = np.column_stack([data_noise_x, data_noise_y])
            data = np.row_stack([data, data_noise])
        # data = data + np.random.normal(scale=0.015, size=data.shape)
        
        circles.append(data)
    circles = np.concatenate(circles, axis=0)
    return circles


def create_disks(n_disks=1, num_pts=100):
    disks = []
    centers = [[0.15, 0.2], [0.75, 0.7], [0.2, .65]]
    for i in range(n_disks):
        # ind = start_from
        np.random.seed(0)
        U1 = np.random.uniform(size=num_pts)
        np.random.seed(42)
        U2 = np.random.uniform(size=num_pts)
        r = 0.2
        data_x = r * np.sqrt(U2) * np.cos(2 * np.pi * U1) + centers[i][0]
        data_y = r * np.sqrt(U2) * np.sin(2 * np.pi * U1) + centers[i][1]
        data = np.column_stack((data_x, data_y))
        disks.append(data)
    disks = np.row_stack(disks)
    return disks


def create_sparse_circles(n_circles=3):
    circles = []
    centers = [[0.8, 0.8], [1.0, 1.0], [0.25, 0.85], [0.6, 0.45]]
    centers = np.array(centers)
    num_points = [50, 90, 35, 20]
    r = [0.15, 0.1, 0.17, 0.3]
    for i in range(n_circles):
        np.random.seed(0)
        theta = np.random.uniform(size=(num_points[i],)) * np.pi * 2
        data_x = r[i] * np.cos(theta).reshape((theta.shape[0], 1)) + centers[i][0]
        data_y = r[i] * np.sin(theta).reshape((theta.shape[0], 1)) + centers[i][1]
        # U1 = np.random.uniform(size=10)
        # U2 = np.random.uniform(size=10)
        # r = 0.1
        # data_noise_x = r * np.sqrt(U2) * np.cos(2 * np.pi * U1) + centers[i][0]
        # data_noise_y = r * np.sqrt(U2) * np.sin(2 * np.pi * U1) + centers[i][1]
        # data_noise = np.column_stack([data_noise_x, data_noise_y])
        data_ = np.concatenate([data_x, data_y], axis=1)
        # data = data + np.random.normal(scale=0.015, size=data.shape)
        # data = np.row_stack([data, data_noise])
        circles.append(data_)
    circles = np.concatenate(circles, axis=0)
    return circles


def create_sparse_disks(n_disks=3):
    disks = []
    centers = [[0.3, 0.2], [0.6, 0.45], [0.25, 0.85], [1.0, 1.0]]
    centers = np.array(centers)
    num_points = [200, 100, 70, 50]
    for i in range(n_disks):
        np.random.seed(0)
        U1 = np.random.uniform(size=num_points[i])
        np.random.seed(0)
        U2 = np.random.uniform(size=num_points[i])
        r = 0.15
        data_x = r * np.sqrt(U2) * np.cos(2 * np.pi * U1) + centers[i][0]
        data_y = r * np.sqrt(U2) * np.sin(2 * np.pi * U1) + centers[i][1]
        data_ = np.column_stack((data_x, data_y))
        disks.append(data_)
    disks = np.row_stack(disks)
    return disks

def create_circles_and_disks(num_circles, num_disks, w_noise=False):
    centers = [[0.3, 0.2], [0.8, 0.8], [0.25, 0.85], [1.0, 0.45]]
    centers = np.array(centers)
    data_pts = []
    num_points = [100, 90, 100, 100]
    r = [0.2, 0.15, 0.17, 0.1]
    for i in range(num_circles):
        if i %2 == 0:
            circle = create_circle(centers[i, :], r[i], w_noise, num_points[i])
        else:
            circle = create_sparse_circles(1)
        data_pts.append(circle)
    for i in range(num_circles, num_circles + num_disks):
        disk = create_disk(centers[i, :], r[i], num_points[i])
        data_pts.append(disk)
    data_ = np.row_stack(data_pts)
    return data_