"""Gradient of the ARAP energy (spokes-and-rims), port of gptoolbox's
arap_gradient.m and the helpers it uses from utils/ARAP.

The matrices depending only on the rest mesh are built once in ArapGradient,
while MATLAB rebuilds them at every call.
"""
import numpy as np
import scipy.sparse as sp

from .mesh import cotmatrix


def cotangent(vert, triv):
    """Half cotangents of the angles at the 3 corners of each triangle."""
    l = np.linalg.norm(vert[triv[:, [1, 2, 0]]] - vert[triv[:, [2, 0, 1]]], axis=2)
    l1, l2, l3 = l.T
    s = (l1 + l2 + l3) / 2
    dbl_a = 2 * np.sqrt(s * (s - l1) * (s - l2) * (s - l3))
    return np.column_stack([l2 ** 2 + l3 ** 2 - l1 ** 2,
                            l1 ** 2 + l3 ** 2 - l2 ** 2,
                            l1 ** 2 + l2 ** 2 - l3 ** 2]) / dbl_a[:, None] / 4


def _spokes_and_rims_block(vert, triv, C, d):
    n = len(vert)
    i1, i2, i3 = triv.T
    c1, c2, c3 = C.T
    x1, x2, x3 = vert[i1, d], vert[i2, d], vert[i3, d]
    I = np.concatenate([i1, i2, i2, i3, i3, i1, i1, i2, i3])
    J = np.concatenate([i2, i1, i3, i2, i1, i3, i1, i2, i3])
    v = np.concatenate([
        c3 * (x1 - x2) + c2 * (x1 - x3),
        -c3 * (x1 - x2) + c1 * (x2 - x3),
        c1 * (x2 - x3) + c3 * (x2 - x1),
        -c1 * (x2 - x3) + c2 * (x3 - x1),
        c2 * (x3 - x1) + c1 * (x3 - x2),
        -c2 * (x3 - x1) + c3 * (x1 - x2),
        # diagonal
        c3 * (x1 - x2) - c2 * (x3 - x1),
        c1 * (x2 - x3) - c3 * (x1 - x2),
        c2 * (x3 - x1) - c1 * (x2 - x3),
    ])
    return sp.csr_matrix((v / 3, (I, J)), shape=(n, n))


def fit_rotations(S):
    """Closest rotation R_i to each covariance S_i (n, 3, 3), without flips."""
    u, _, vt = np.linalg.svd(S)
    flip = np.linalg.det(vt.transpose(0, 2, 1) @ u.transpose(0, 2, 1)) < 0
    u[flip, :, -1] *= -1
    return vt.transpose(0, 2, 1) @ u.transpose(0, 2, 1)


class ArapGradient:
    """ARAP gradient with respect to the rest mesh (vert, triv)."""

    def __init__(self, vert, triv):
        C = cotangent(vert, triv)
        self.L = cotmatrix(vert, triv)
        self.K = [_spokes_and_rims_block(vert, triv, C, d) for d in range(3)]
        self.KT = [k.T.tocsr() for k in self.K]
        # weighted squared length of the rest edges of every cell
        self.rest_norm = np.trace(self.covariances(vert), axis1=1, axis2=2)

    def covariances(self, U):
        """Per-vertex covariance S_i[a, b] = (K_a' U)[i, b]."""
        return np.stack([kt @ U for kt in self.KT], axis=1)

    def rotations(self, U):
        """Best rotation at every vertex for the positions U (local step)."""
        return fit_rotations(self.covariances(U))

    def similarities(self, U):
        """Best rotation times scale at every vertex (local step of ASAP)."""
        S = self.covariances(U)
        R = fit_rotations(S)
        s = np.trace(R @ S, axis1=1, axis2=2) / self.rest_norm
        return np.maximum(s, 0)[:, None, None] * R

    def rhs(self, R):
        """With the rotations R fixed, the gradient is -(L U + rhs(R))."""
        return sum(self.K[c] @ R[:, :, c] for c in range(3))

    def __call__(self, U):
        return -(self.L @ U + self.rhs(self.rotations(U)))
