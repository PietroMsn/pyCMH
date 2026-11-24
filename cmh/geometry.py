"""Mesh loading, areas and Laplace-Beltrami basis.

Python port of utils/getShape.m, utils/load_shape.m, UtilsFM/Fmap/* and
utils_CMH/scaling_shapes.m.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from .mesh import cotmatrix, read_triangle_mesh

DATA = Path(__file__).resolve().parent / 'data'


@dataclass
class Shape:
    vert: np.ndarray          # (n, 3) vertex coordinates
    triv: np.ndarray          # (m, 3) 0-based triangle indices
    A: sp.csr_matrix          # lumped mass matrix with the true areas
    Ae: sp.csr_matrix         # lumped mass matrix normalized to total area 1
    landmarks: np.ndarray     # 0-based landmark indices


@dataclass
class LBBasis:
    evecs: np.ndarray         # (n, k) eigenfunctions, Ae-orthonormal
    evals: np.ndarray         # (k,) eigenvalues in ascending order
    A: sp.csr_matrix          # normalized mass matrix
    W: sp.csr_matrix          # stiffness matrix (positive semi-definite)


def load_shape(path):
    """Load a mesh given its path without extension (.off, .obj, .ply or .mat)."""
    path = Path(path)
    for ext in ('.off', '.ply', '.obj'):
        if path.with_suffix(ext).exists():
            vert, triv = read_triangle_mesh(str(path.with_suffix(ext)))
            return vert.astype(float), triv.astype(np.int64)

    mat = sio.loadmat(str(path.with_suffix('.mat')), squeeze_me=True, struct_as_record=False)
    if 'Src' in mat:
        vert, triv = mat['Src'].X, mat['Src'].T
    elif 'shape' in mat or 'surface' in mat:
        s = mat['shape'] if 'shape' in mat else mat['surface']
        vert, triv = np.column_stack([s.X, s.Y, s.Z]), s.TRIV
    else:
        raise ValueError(f'{path}.mat has no Src, shape or surface field')
    return np.asarray(vert, dtype=float), np.asarray(triv, dtype=np.int64) - 1


def tri_areas(vert, triv):
    e1 = vert[triv[:, 2]] - vert[triv[:, 0]]
    e2 = vert[triv[:, 1]] - vert[triv[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)


def mass_matrix(vert, triv):
    """Lumped mass matrix: each vertex gets 1/3 of the area of its triangles."""
    a = np.bincount(triv.ravel(), weights=np.repeat(tri_areas(vert, triv) / 3, 3),
                    minlength=len(vert))
    return sp.diags(a).tocsr()


LAPLACIANS = ('cotan', 'robust', 'robust_pc')


def laplacian(vert, triv, kind='cotan'):
    """Stiffness matrix (positive semi-definite) and lumped mass matrix.

    cotan: the cotangent Laplacian of lb_basis_surface.m; robust: the tufted
    intrinsic Delaunay Laplacian of Sharp & Crane (2020), which handles
    non-manifold and badly shaped triangles but keeps the connected components
    apart; robust_pc: their point cloud Laplacian on the vertices, which
    ignores the connectivity and connects nearby components."""
    if kind == 'cotan':
        return -cotmatrix(vert, triv), mass_matrix(vert, triv)
    import robust_laplacian
    if kind == 'robust':
        W, M = robust_laplacian.mesh_laplacian(np.asarray(vert, float), np.asarray(triv))
    elif kind == 'robust_pc':
        W, M = robust_laplacian.point_cloud_laplacian(np.asarray(vert, float))
    else:
        raise ValueError(f'unknown Laplacian {kind}')
    return W.tocsr(), M.tocsr()


def lb_basis(vert, triv, k, kind='cotan'):
    """First k eigenpairs of the Laplacian `kind` (see `laplacian`), as
    lb_basis_surface.m for the cotangent one."""
    W, M = laplacian(vert, triv, kind)
    a = M.diagonal()
    A = sp.diags(a / a.sum()).tocsr()

    evals, evecs = eigsh(W, k=k, M=A, sigma=-1e-5)
    order = np.argsort(evals)
    return LBBasis(evecs=evecs[:, order], evals=evals[order], A=A, W=W)


def get_shape(path, landmarks, k=200):
    return make_shape(*load_shape(path), landmarks, k)


def make_shape(vert, triv, landmarks, k=200):
    L = lb_basis(vert, triv, k)
    shape = Shape(vert=vert, triv=triv, A=mass_matrix(vert, triv), Ae=L.A,
                  landmarks=np.asarray(landmarks))
    return shape, L


def scale_to_smpl(M):
    """Rescale the shape to the total area of the SMPL template.

    As in the MATLAB code, M.A keeps the areas before the scaling."""
    scaling_w = sio.loadmat(str(DATA / 'scaling_weights.mat'))['scaling_w']
    M.vert = np.sqrt(scaling_w.sum() / M.A.sum()) * M.vert
    return M


def scaling_shapes(M1, M2):
    return scale_to_smpl(M1), scale_to_smpl(M2)
