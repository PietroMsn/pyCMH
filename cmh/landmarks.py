"""Landmarks for the wave kernel map descriptors: perturbation of the given
ones, to measure the dependence of the pipeline on them, and automatic
landmarks at the extremities of the shapes.

The automatic landmarks are the local maxima of the average geodesic distance
(AGD), the tips of the limbs, head, tail... For a pair of shapes they are
matched by the consistency of their geodesic distances; the intrinsic cost
cannot tell the left and right hands (or feet) apart, so among the best
assignments we take the one whose map is locally the most coherent rigid
motion: swapping only the hands or only the feet twists the body.
"""
from itertools import combinations, permutations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components, dijkstra

from .descriptors import _wks_filters
from .fmap import icp_refine
from .refine import nearest_neighbors
from .geometry import mass_matrix
from .mesh import adjacency_matrix, edges


def heat_geodesics(vert, triv, sources):
    """Geodesic distances from every source, as (n_sources, n), with the heat
    method on the intrinsic Delaunay tufted cover of Sharp & Crane (2020).

    The heat method of libigl on the original triangles is wrong by up to 0.8
    sqrt(area) on some SHREC'19 scans and SMAL meshes. Vertices in another
    connected component than the source get the largest finite distance."""
    import potpourri3d as pp3d
    solver = pp3d.MeshHeatMethodDistanceSolver(np.asarray(vert, float), np.asarray(triv), use_robust=True)
    D = np.stack([solver.compute_distance(int(s)) for s in sources])
    _, comp = connected_components(adjacency_matrix(triv, len(vert)), directed=False)
    far = comp[None, :] != comp[np.asarray(sources)][:, None]
    D[far | ~np.isfinite(D)] = np.nan
    return np.where(np.isnan(D), np.nanmax(D, axis=1, keepdims=True), D)


def perturb_landmarks(vert, triv, landmarks, radius, seed=0):
    """Move every landmark to a random vertex at geodesic distance radius *
    sqrt(area) from it (within 10%), measured on the edge graph: the heat
    method is not reliable on some of the SHREC'19 scans."""
    rng = np.random.default_rng(seed)
    d = dijkstra(edge_graph(vert, triv), directed=False, indices=landmarks) / np.sqrt(mass_matrix(vert, triv).sum())
    out = []
    for row in d:
        ring = np.flatnonzero(np.abs(row - radius) < 0.1 * radius)
        out.append(rng.choice(ring) if len(ring) else np.argmin(np.abs(row - radius)))
    return np.array(out)


def edge_graph(vert, triv):
    """Edge graph of the mesh weighted by the edge lengths."""
    e = edges(triv)
    w = np.linalg.norm(vert[e[:, 0]] - vert[e[:, 1]], axis=1)
    G = sp.coo_matrix((w, (e[:, 0], e[:, 1])), shape=(len(vert),) * 2)
    return (G + G.T).tocsr()


class Extremities:
    """Candidate landmarks of a shape and the quantities used to match them.

    agd: average geodesic distance of every vertex (over sqrt(area)), smoothed
    in the first k_smooth LB eigenfunctions; candidates: the vertices of the
    largest component that are maxima of agd within radius * sqrt(area),
    sorted by decreasing agd; dist: geodesic distances between them.
    """

    def __init__(self, vert, triv, evecs, A, n_candidates=8, radius=0.15, n_sources=64,
                 k_smooth=30, seed=0):
        scale = np.sqrt(mass_matrix(vert, triv).sum())
        G = edge_graph(vert, triv)
        _, comp = connected_components(G, directed=False)
        main = np.flatnonzero(comp == np.bincount(comp).argmax())

        sources = np.random.default_rng(seed).choice(main, min(n_sources, len(main)), replace=False)
        D = dijkstra(G, directed=False, indices=sources)[:, main] / scale
        agd = np.full(len(vert), -np.inf)
        agd[main] = D.mean(axis=0)
        phi = evecs[:, :k_smooth]
        smooth = phi @ (phi.T @ (A @ np.where(np.isfinite(agd), agd, 0)))
        agd[main] = smooth[main]
        self.agd = agd

        # local maxima on the 1-ring, then within the geodesic radius
        adj = (G > 0).tocsr()
        ring_max = np.array([agd[v] >= agd[adj.indices[adj.indptr[v]:adj.indptr[v + 1]]].max()
                             for v in main])
        cand = main[ring_max]
        cand = cand[np.argsort(-agd[cand])]
        keep = []
        for v in cand:
            d = dijkstra(G, directed=False, indices=v, limit=radius * scale)
            if agd[v] >= agd[np.isfinite(d)].max():
                keep.append(v)
            if len(keep) == n_candidates:
                break
        self.candidates = np.array(keep)

        dc = dijkstra(G, directed=False, indices=self.candidates) / scale
        self.dist = dc[:, self.candidates]


def local_rotations(X, Y):
    """Proper rotation R minimizing |(X - mean) R^T - (Y - mean)| (Kabsch)."""
    U, _, Vt = np.linalg.svd((X - X.mean(0)).T @ (Y - Y.mean(0)))
    return (U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T


def random_patches(M, n_patches=200, radius=0.1, seed=0):
    """Vertices of n_patches Euclidean balls of radius * sqrt(area) around
    random vertices of M (the balls with fewer than 10 vertices are dropped)."""
    centers = np.random.default_rng(seed).choice(len(M.vert), min(n_patches, len(M.vert)), replace=False)
    r = radius * np.sqrt(mass_matrix(M.vert, M.triv).sum())
    return [b for b in map(np.array, cKDTree(M.vert).query_ball_point(M.vert[centers], r)) if len(b) >= 10]


def rotation_incoherence(M1, M2, p2p, patches):
    """Mean angle between the rigid rotations that best align the patches of
    M2 (random_patches) to their images on M1 through p2p (vertex of M2 ->
    vertex of M1). Between two poses the rotations of nearby parts differ by
    the joint angles; a map with the hands or the feet swapped twists a part
    of the body by 180 degrees, and a mirrored map has no coherent proper
    rotation at all."""
    R = np.stack([local_rotations(M2.vert[b], M1.vert[p2p[b]]) for b in patches])
    cos = (np.einsum('aij,bij->ab', R, R) - 1) / 2
    return float(np.arccos(np.clip(cos, -1, 1)).mean())


def landmark_hypotheses(E1, E2, n_landmarks=5, n_hypotheses=4, relative=False):
    """The n_hypotheses best ordered landmarks (n_landmarks, 2) on the two
    shapes. Every injective assignment of n_landmarks candidates of E1 to
    candidates of E2 is scored by the mean difference of the geodesic
    distances between the landmarks and of their agd; with relative, both
    are divided by their mean over the assignment, so that shapes of
    different proportions (a cat and a horse) can match. The cost is almost
    invariant to the symmetries of the shapes: for a human, the 4 best
    assignments are the ones with the hands and the feet swapped or not."""
    m1, m2 = len(E1.candidates), len(E2.candidates)
    n = min(n_landmarks, m1, m2)
    sub1 = np.array(list(combinations(range(m1), n)))
    sub2 = np.array(list(permutations(range(m2), n)))
    iu = np.triu_indices(n, 1)
    rel = (lambda x: x / x.mean(axis=1, keepdims=True)) if relative else (lambda x: x)
    d1 = rel(np.stack([E1.dist[np.ix_(s, s)][iu] for s in sub1]))   # (S1, pairs)
    d2 = rel(np.stack([E2.dist[np.ix_(s, s)][iu] for s in sub2]))   # (S2, pairs)
    a1, a2 = rel(E1.agd[E1.candidates][sub1]), rel(E2.agd[E2.candidates][sub2])
    cost = (np.abs(d1[:, None] - d2[None]).mean(axis=2)
            + np.abs(a1[:, None] - a2[None]).mean(axis=2))           # (S1, S2)
    best = np.argsort(cost, axis=None)[:n_hypotheses]
    return [np.column_stack([E1.candidates[sub1[a]], E2.candidates[sub2[b]]])
            for a, b in zip(*np.unravel_index(best, cost.shape))]


def quick_map(L1, L2, landmarks, query, k=30, num_times=200, skip_size=10, mu=1e-3, n_icp=5,
              n_samples=3000, seed=0):
    """Cheap vertex map (vertex of M2 -> vertex of M1) for a landmark
    hypothesis: functional map in the first k LB eigenfunctions from the WKS
    and the wave kernel maps of the landmarks (the descriptors of the
    pipeline), in closed form with a diagonal Laplacian commutativity
    penalty, refined with ICP. The descriptors are taken in the spectral
    domain: the basis is A-orthonormal, so their coefficients are exact.
    ICP matches n_samples random vertices of each shape, and the map is
    returned on the vertices query of M2 only (-1 elsewhere)."""
    def coefficients(L, lm):
        T = _wks_filters(L.evals, num_times)
        wks = (L.evecs.T @ (L.A @ L.evecs ** 2)) @ T
        F = np.column_stack([wks] + [T * L.evecs[v][:, None] for v in lm])[:, ::skip_size]
        return F
    F1, F2 = coefficients(L1, landmarks[:, 0]), coefficients(L2, landmarks[:, 1])
    no = np.linalg.norm(F1, axis=0)
    F1, F2 = F1[:k] / no, F2[:k] / no
    pen = (L2.evals[:k, None] - L1.evals[None, :k]) ** 2
    pen = mu * pen / pen.max()
    G = F1 @ F1.T
    C = np.stack([np.linalg.solve(G + np.diag(pen[i] * np.trace(G)), F1 @ F2[i]) for i in range(k)])
    rng = np.random.default_rng(seed)
    s1 = rng.choice(len(L1.evecs), min(n_samples, len(L1.evecs)), replace=False)
    s2 = rng.choice(len(L2.evecs), min(n_samples, len(L2.evecs)), replace=False)
    phi1 = L1.evecs[:, :k]
    C = icp_refine(phi1[s1], L2.evecs[s2, :k], C, n_icp)[0]
    p2p = np.full(len(L2.evecs), -1)
    p2p[query] = nearest_neighbors(phi1 @ C.T, L2.evecs[query, :k])
    return p2p


def match_extremities(M1, L1, E1, M2, L2, E2, map_fn, n_best=4, n_relative=32, n_quick=3):
    """Landmarks (n_landmarks, 2) of the hypothesis whose vertex map
    map_fn(landmarks) (vertex of M2 -> vertex of M1) has the most coherent
    local rotations, and that map. The hypotheses mapped with map_fn are the
    n_best of the geodesic cost, and the n_quick whose quick_map is the most
    coherent among the n_relative best of the relative cost: the first are
    enough for humans, the second find the assignments of quadrupeds of
    different proportions, where the four paws make the cost ambiguous."""
    patches = random_patches(M2)
    query = np.unique(np.concatenate(patches))
    hyps = landmark_hypotheses(E1, E2, n_hypotheses=n_best)
    rel = landmark_hypotheses(E1, E2, n_hypotheses=n_relative, relative=True)
    quick = [rotation_incoherence(M1, M2, quick_map(L1, L2, lm, query), patches) for lm in rel]
    for h in np.argsort(quick)[:n_quick]:
        if not any(np.array_equal(rel[h], lm) for lm in hyps):
            hyps.append(rel[h])
    best = None
    for lm in hyps:
        p2p = map_fn(lm)
        score = rotation_incoherence(M1, M2, p2p, patches)
        if best is None or score < best[0]:
            best = score, lm, p2p
    return best[1], best[2]
