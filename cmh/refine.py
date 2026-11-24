"""Refinement of functional maps: ZoomOut [Melzi et al. 2019] in the CMH
basis, and its bidirectional variant with a bijectivity term.

Conventions as in cmh.fmap: C (k2, k1) maps coefficients on shape 1 to shape
2, p2p[v] is the vertex of shape 1 matched to vertex v of shape 2.

On large meshes the iterations only match random subsets of vertices, as in
the original ZoomOut, and the full vertex map is computed at the end.
"""
from dataclasses import dataclass

import numpy as np

from .basis import compute_cmh


@dataclass
class LabelCompat:
    """Soft constraint on the vertex maps between two labelled shapes: a match
    between a vertex of class c2 of shape 2 and one of class c1 of shape 1
    pays forbidden[c2, c1] * weight * (mean squared distance of the unconstrained
    matches), so the weight does not depend on the size of the basis."""
    class1: np.ndarray      # class of every vertex of shape 1
    class2: np.ndarray      # class of every vertex of shape 2
    forbidden: np.ndarray   # (classes of shape 2, classes of shape 1), 0 or 1
    weight: float = 1.0


def nearest_neighbors(X, Y, block_size=2 ** 25, penalty=None):
    """Index of the nearest row of X for each row of Y (brute force, which is
    faster than a k-d tree in the high dimensions of spectral embeddings).
    Distances are computed by blocks of at most block_size entries. penalty is
    (classes of the rows of X, classes of the rows of Y, table (classes of Y,
    classes of X), weight), see LabelCompat."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    x2 = np.einsum('ij,ij->i', X, X)
    block = max(1, block_size // len(X))
    out = np.empty(len(Y), dtype=np.int64)
    scale = None
    for s in range(0, len(Y), block):
        Yb = np.asarray(Y[s:s + block], dtype=np.float32)
        D = x2[None, :] - 2 * Yb @ X.T
        if penalty is not None:
            cx, cy, table, weight = penalty
            if scale is None:   # from the first block
                scale = float(np.mean(D.min(axis=1) + np.einsum('ij,ij->i', Yb, Yb)))
            D += (weight * scale * table[cy[s:s + block]][:, cx]).astype(np.float32)
        out[s:s + block] = np.argmin(D, axis=1)
    return out


def p2p_from_fmap(phi1, phi2, C):
    return nearest_neighbors(phi1 @ C.T, phi2)


def fmap_from_p2p(phi1, phi2, a2, p2p):
    """Least squares functional map given the vertex map (phi2 is A2-orthonormal)."""
    return phi2.T @ (a2[:, None] * phi1[p2p])


def spectral_basis(shape, L, k, cmh):
    """First k LB eigenfunctions, followed by the 3 CMH functions if cmh."""
    phi = L.evecs[:, :k]
    return compute_cmh(shape, phi) if cmh else phi


def _samples(n, n_samples, seed=0):
    if n_samples is None or n <= n_samples:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, n_samples, replace=False))


def _fmap(phi1, phi2, a2, s2, matched):
    """Functional map from the matches `matched` (vertices of shape 1) of the
    samples s2 of shape 2."""
    if len(s2) == len(phi2):
        return fmap_from_p2p(phi1, phi2, a2, matched)
    return np.linalg.lstsq(phi2[s2], phi1[matched], rcond=None)[0]


def zoomout(M1, L1, M2, L2, p2p, k_start, k_end, step=1, cmh=True, n_samples=None):
    """ZoomOut from k_start to k_end eigenfunctions, starting from p2p.

    With cmh=True the basis at every level is the LB basis of that size
    extended with its 3 CMH functions."""
    s1, s2 = _samples(len(M1.vert), n_samples), _samples(len(M2.vert), n_samples)
    a2 = L2.A.diagonal()
    matched = p2p[s2]
    for k in range(k_start, k_end + 1, step):
        phi1, phi2 = spectral_basis(M1, L1, k, cmh), spectral_basis(M2, L2, k, cmh)
        C = _fmap(phi1, phi2, a2, s2, matched)
        matched = s1[p2p_from_fmap(phi1[s1], phi2[s2], C)]
    return C, p2p_from_fmap(phi1, phi2, C)


def bijective_zoomout(M1, L1, M2, L2, p2p21, p2p12, k_start, k_end, step=1, cmh=True,
                      w=1.0, n_samples=None, both=False, compat=None):
    """ZoomOut on both directions coupled by a bijectivity term, in the
    spirit of the discrete optimization of Ren et al. [2021].

    Each vertex map is recovered from an embedding that asks both for
    C21 phi2 ~ phi1 and for consistency with the map in the other direction
    (C12 C21 ~ I), to reduce the many-to-one matches of plain ZoomOut.
    Returns the map from shape 2 to shape 1 and, with both, also the map from
    shape 1 to shape 2. compat (LabelCompat) penalizes the matches between
    incompatible labels in both directions.
    """
    s1, s2 = _samples(len(M1.vert), n_samples), _samples(len(M2.vert), n_samples)
    a1, a2 = L1.A.diagonal(), L2.A.diagonal()
    m21, m12 = p2p21[s2], p2p12[s1]

    def pen21(i1, i2):
        return None if compat is None else (compat.class1[i1], compat.class2[i2], compat.forbidden, compat.weight)

    def pen12(i1, i2):
        return None if compat is None else (compat.class2[i2], compat.class1[i1], compat.forbidden.T, compat.weight)

    for k in range(k_start, k_end + 1, step):
        phi1, phi2 = spectral_basis(M1, L1, k, cmh), spectral_basis(M2, L2, k, cmh)
        C21 = _fmap(phi1, phi2, a2, s2, m21)   # coefficients on shape 1 -> shape 2
        C12 = _fmap(phi2, phi1, a1, s1, m12)   # coefficients on shape 2 -> shape 1
        m21 = s1[nearest_neighbors(np.hstack([phi1[s1] @ C21.T, w * phi1[s1]]),
                                   np.hstack([phi2[s2], w * phi2[s2] @ C12.T]), penalty=pen21(s1, s2))]
        m12 = s2[nearest_neighbors(np.hstack([phi2[s2] @ C12.T, w * phi2[s2]]),
                                   np.hstack([phi1[s1], w * phi1[s1] @ C21.T]), penalty=pen12(s1, s2))]
    all1, all2 = np.arange(len(phi1)), np.arange(len(phi2))
    p21 = nearest_neighbors(np.hstack([phi1 @ C21.T, w * phi1]),
                            np.hstack([phi2, w * phi2 @ C12.T]), penalty=pen21(all1, all2))
    if not both:
        return p21
    return p21, nearest_neighbors(np.hstack([phi2 @ C12.T, w * phi2]),
                                  np.hstack([phi1, w * phi1 @ C21.T]), penalty=pen12(all1, all2))
