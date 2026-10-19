"""Profile residual for independent scale fitting to dictionary atoms."""
import numpy as np

def projection_residual(y, D, Dn=None):
    """||y - beta*d|| with beta fitted per atom: the magnitude-nuisance profile residual."""
    Dn = np.linalg.norm(D, axis=1) if Dn is None else Dn
    b = D @ y / np.maximum(Dn ** 2, 1e-18)
    r = np.linalg.norm(y[None, :] - b[:, None] * D, axis=1)
    r[Dn == 0] = np.inf
    return r
