import torch
import pickle as pck
import numpy as np
from xgboost import XGBClassifier
from torch_geometric.datasets import TUDataset
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
import argparse

"""
Edited by Anonymous (will be updated after review) for the experiments.
Original Author: Mathieu Carrière
This part of code is from https://github.com/MathieuCarriere/multipers/blob/main/experiments.ipynb
"""
class MultiPersistenceLandscapeWrapper(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn wrapper for cross-validating Multiparameter Persistence Landscapes.
    """
    def __init__(self, power=0, step=1, k=None):
        self.power, self.step, self.k = power, step, k

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        final = []
        for nf in range(len(X[0])):
            XX = [X[idx][nf] for idx in range(len(X))]
            if self.k is None:
                Y = np.vstack([  np.maximum.reduce([np.multiply(im, np.reshape(w**self.power, [1,1,-1])).sum(axis=2).flatten()[np.newaxis,:] for [im,w] in L])  for L in XX  ])
            else:
                Y = np.vstack([  LS[:,:,:self.k].sum(axis=2).flatten()[np.newaxis,:] for LS in XX  ])
            res = int(np.sqrt(Y.shape[1]))
            nr = int(res/self.step)	
            Y = np.reshape(np.transpose(np.reshape(np.transpose(np.reshape(Y,[-1,res,nr,self.step]).sum(axis=3),(0,2,1)),[-1,nr,nr,self.step]).sum(axis=3),(0,2,1)),[-1,nr**2])
            final.append(Y)
        return np.hstack(final)

def save_as_pickle(dataset, num_samples, l):
    data_dir = f"./graph_landscapes/{dataset}/landscape_values_hks_l_{l}"
    data_h0 = []
    data_h1 = []
    for i in range(num_samples):
        fname = f"{data_dir}/graph_{i}.pt"
        data = torch.load(fname)
        vec_0 = data[f'l_{l}_H_0'].numpy()
        vec_1 = data[f'l_{l}_H_1'].numpy()
        num_div = int(np.sqrt(vec_0.shape[0]))
        vec_0 = vec_0.reshape((num_div, num_div, vec_0.shape[1]))
        vec_1 = vec_1.reshape((num_div, num_div, vec_1.shape[1]))
        # vec_0 = vec_0[1:, 1:, :]
        # vec_1 = vec_1[1:, 1:, :]
        data_h0.append(vec_0)
        data_h1.append(vec_1)
    save_dir = f"./graph_landscapes/{dataset}"
    save_name = f"{save_dir}/mls_l_{l}_HKS-RC-0.pkl"
    pck.dump(data_h0, open(save_name, "wb"))
    print(f"Saved to {save_name}")
    save_name = f"{save_dir}/mls_l_{l}_HKS-RC-1.pkl"
    pck.dump(data_h1, open(save_name, "wb"))
    print(f"Saved to {save_name}", flush=True)
    

def classify(ds_name, data, Xmls, **kwargs):
    classifier = XGBClassifier(random_state=1)
    cv = 5
    labels = np.array([d.y.item() for i, d in enumerate(data)])
    npoints = len(labels)
    train_index, test_index = train_test_split(np.arange(npoints), test_size=0.2, stratify=labels)
    params_mls = {
        "mls__power":   [0, 1],
        "mls__step":    [1, 5],
        "mls__k":       [5],
        "clf":          [classifier],
    }
    pipe_mls = Pipeline([("mls", MultiPersistenceLandscapeWrapper()), ("clf", classifier)])
    X_train  = [[Xmls[nf][n] for nf in range(len(Xmls))] for n in train_index]
    X_test   =[[Xmls[nf][n] for nf in range(len(Xmls))] for n in test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    model = GridSearchCV(estimator=pipe_mls, param_grid=params_mls, cv=cv, verbose=0, n_jobs = -1)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='PROTEINS', type=str, choices=['PROTEINS','IMDB-BINARY', 'MUTAG'])
    parser.add_argument("--l", default=[2], nargs='+', type=int)
    args = parser.parse_args()
    dataset = args.dataset
    l = args.l
    
    path = f"./graph_landscapes/{dataset}/"
    list_filts = ["HKS-RC-0", "HKS-RC-1"]
    data = TUDataset("./data", name=dataset) 
    
    num_samples = len(data)  
    
    print(f"***** {data} l = {l} *****", flush=True)
    
    
    num_exps = 5
    Xmls = [pck.load(open(f"{path}/mls_l_{ell}_{filt}.pkl", "rb")) for ell in l for filt in list_filts]
    np.random.seed(42)
    scores = []
    for i in range(1, num_exps + 1):
        score = classify(dataset, data, Xmls)
        scores.append(score)
        print(f"Fold {i}: Acc: {score}", flush=True)
    print(f"Final acc: {np.mean(scores)} std: {np.std(scores)}", flush=True)