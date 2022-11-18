import numpy as np

def calc(df, r):
    x = df - df.iloc[:, :].mean()
    normalized = x / df.std()
    X = normalized.to_numpy()
    U,D,Vt = np.linalg.svd(X, full_matrices=False)
    pc = Vt.T[:, :r].T
    a = U@np.diag(D)[:, :r]
    std = D / np.sqrt(len(X) - 1)
    return X, U, D, Vt, pc, a, std


def calc2(data, r):
    x = data - np.mean(data)
    normalized = x / np.std(data)
    X = normalized
    U,D,Vt = np.linalg.svd(X, full_matrices=False)
    pc = Vt.T[:, :r].T
    a = U@np.diag(D)[:, :r]
    std = D / np.sqrt(len(X) - 1)
    return X, U, D, Vt, pc, a, std

def get_dimensions(eigenvalues, error):
    for dim in range(len(eigenvalues)):
        if 1 - (np.sum(eigenvalues[:dim]) / np.sum(eigenvalues)) <= error:
            break
    return dim