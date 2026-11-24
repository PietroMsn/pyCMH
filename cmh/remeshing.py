"""Functional remeshing, port of perform_functional_remeshing.m and
utils/arap_dato.m, and its local-global variant."""
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg, splu

from .arap import ArapGradient
from .mesh import per_face_normals, per_vertex_normals, point_mesh_squared_distance


def _closest_points(vert, triv, query):
    return point_mesh_squared_distance(query, vert, triv)[2]


def arap_dato(new, M1, M2, delta_t, a_arap, n_iter=399):
    """ARAP flow towards M2 plus attraction to the surface of M1."""
    arap_grad = ArapGradient(M2.vert, M2.triv)
    surface_points = _closest_points(M1.vert, M1.triv, new)
    for i in range(1, n_iter + 1):
        if i % 50 == 0:
            print(f'iter:{i}')
            surface_points = _closest_points(M1.vert, M1.triv, new)
        new = new - delta_t * (a_arap * arap_grad(new) + (new - surface_points))
    return new


def arap_local_global(new, M1, M2, a_arap, data='plane', beta=0.1, gamma=0.0, asap=False,
                      max_iter=100, tol=1e-4, weights=None, min_cos=None, max_dist=None):
    """Minimizes the energy of arap_dato, a_arap * ARAP(M2) + data term
    towards the surface of M1, with the local-global iterations of Sorkine and
    Alexa instead of explicit gradient steps.

    Every iteration updates the closest points p on M1 and the rotations, then
    solves for the positions. The data term is |u - p|^2 / 2 (data='point',
    the energy of arap_dato) or ((n.(u - p))^2 + beta |u - p|^2) / 2 with n the
    normal of M1 at p (data='plane'), which lets the vertices slide on the
    surface. gamma > 0 adds gamma |u - u0|^2 / 2, a weak attraction to the
    initial positions given by the map, against the drift along the surface.
    With asap the local step fits a rotation and a scale for every cell
    (as-similar-as-possible), so that the rest shape M2 can change its
    proportions to the ones of M1 without an ARAP penalty. weights (n,)
    multiply the data term of every vertex: with weight 0 on the vertices of
    M2 hidden under a garment of M1, they follow the ARAP term only, instead
    of being pulled onto the garment. With min_cos and max_dist the closest
    points are rejected at every iteration if the normal of M1 there and the
    current normal of the vertex form an angle with cosine below min_cos, or
    if they are farther than max_dist (as in non-rigid ICP): the vertices of
    an arm do not latch onto the torso they touch. Stops when the vertices move by less
    than tol times the size of the shape. Returns the positions and the
    number of iterations.
    """
    arap = ArapGradient(M2.vert, M2.triv)
    H = (-a_arap * arap.L).tocsc()
    w0 = np.ones(len(new)) if weights is None else np.asarray(weights, float)
    w = w0
    W = sp.diags(w).tocsc()
    eye = sp.identity(len(new), format='csc')
    face_normals = per_face_normals(M1.vert, M1.triv)
    # with data='plane' the matrix changes at every iteration: conjugate
    # gradient, preconditioned with the average of the data term
    lu = splu(H + (1 if data == 'point' else beta + 1 / 3) * W + gamma * eye)
    anchor = gamma * new
    size = np.linalg.norm(new - new.mean(axis=0)) / np.sqrt(len(new))

    for it in range(1, max_iter + 1):
        sq, face, p = point_mesh_squared_distance(new, M1.vert, M1.triv)
        if min_cos is not None or max_dist is not None:
            ok = np.ones(len(new), bool)
            if min_cos is not None:
                n2 = per_vertex_normals(new, M2.triv)
                ok &= np.sum(face_normals[face] * n2, axis=1) >= min_cos
            if max_dist is not None:
                ok &= sq <= max_dist ** 2
            w = w0 * ok
            if data == 'point':
                lu = splu(H + sp.diags(w).tocsc() + gamma * eye)
        b = a_arap * arap.rhs(arap.similarities(new) if asap else arap.rotations(new))
        if data == 'point':
            U = lu.solve(b + w[:, None] * p + anchor)
        else:
            nrm = face_normals[face]
            D = lambda X: w[:, None] * (nrm * np.sum(nrm * X, axis=1, keepdims=True) + beta * X)
            G = lambda X: D(X) + gamma * X
            shape = new.shape
            A = LinearOperator((new.size, new.size), dtype=float,
                               matvec=lambda x: (H @ x.reshape(shape) + G(x.reshape(shape))).ravel())
            P = LinearOperator((new.size, new.size), dtype=float,
                               matvec=lambda x: lu.solve(x.reshape(shape)).ravel())
            U = cg(A, (b + D(p) + anchor).ravel(), x0=new.ravel(), rtol=1e-8, maxiter=500, M=P)[0].reshape(shape)
        step = np.linalg.norm(U - new) / np.sqrt(len(new))
        new = U
        if step < tol * size:
            break
    return new, it


def arap_targets(new, M2, targets, weights, a_arap, max_iter=100, tol=1e-4):
    """Minimizes a_arap * ARAP(M2) + sum_i weights_i |u_i - targets_i|^2 / 2
    with local-global iterations from the positions new: the vertices follow
    fixed targets (e.g. the points matched by a map) instead of the closest
    points of a surface, so they cannot slide, and the vertices with weight 0
    only follow the ARAP term. Returns the positions and the number of
    iterations."""
    arap = ArapGradient(M2.vert, M2.triv)
    w = np.asarray(weights, float)
    lu = splu((-a_arap * arap.L + sp.diags(w)).tocsc())
    size = np.linalg.norm(new - new.mean(axis=0)) / np.sqrt(len(new))
    for it in range(1, max_iter + 1):
        U = lu.solve(a_arap * arap.rhs(arap.rotations(new)) + w[:, None] * targets)
        step = np.linalg.norm(U - new) / np.sqrt(len(new))
        new = U
        if step < tol * size:
            break
    return new, it


def perform_functional_remeshing(M1, M2, landmarks, p2p, hands_radius, a_arap,
                                 delta_t, cpd_on=False, solver='explicit'):
    """Geometry of M1 with the connectivity of M2.

    Returns the vertices after the ARAP flow, the final vertices (both to use
    with M2.triv) and the time spent in each step. solver='explicit' runs the
    fixed gradient steps of the MATLAB code; 'local_global' replaces both
    phases with arap_local_global and point-to-plane data term.
    """
    times = {}

    t = time.perf_counter()
    if cpd_on:
        from .local_patch import local_patch_refine   # optional: needs libigl and pycpd
        print('Applying Coherent Point Drift on local patches...')
        p2p = local_patch_refine(M1, M2, landmarks[1], p2p, hands_radius)
        p2p = local_patch_refine(M1, M2, landmarks[4], p2p, hands_radius)
    new = M1.vert[p2p]
    times['cpd'] = time.perf_counter() - t
    print(f"done in {times['cpd']:f} seconds")

    if solver == 'local_global':
        t = time.perf_counter()
        new2, _ = arap_local_global(new, M1, M2, a_arap)
        times['arap_dato'] = time.perf_counter() - t
        return new, new2, times

    print('processing arap for the mesh consistency...')
    t = time.perf_counter()
    arap_grad = ArapGradient(M2.vert, M2.triv)
    for _ in range(200):
        new = new - delta_t * a_arap * arap_grad(new)
    times['arap'] = time.perf_counter() - t
    print(f"done in {times['arap']:f} seconds")

    print('Minimizing the ARAP energy to fit the original data...')
    t = time.perf_counter()
    new2 = arap_dato(new, M1, M2, delta_t, a_arap)
    times['arap_dato'] = time.perf_counter() - t
    print(f"done in {times['arap_dato']:f} seconds")

    return new, new2, times
