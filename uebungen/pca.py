import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calc(df, r):
    x = df - df.iloc[:, :].mean()
    normalized = x / df.std()
    X = normalized.to_numpy()
    U,D,Vt = np.linalg.svd(X, full_matrices=False)
    pc = Vt.T[:, :r]
    a = U@np.diag(D)[:, :r]
    std = D / np.sqrt(len(X) - 1)
    return X, U, D, Vt, pc, a, std