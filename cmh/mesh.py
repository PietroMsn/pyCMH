"""Triangle-mesh utilities on numpy and scipy only, in place of the libigl
functions the package used: reading and writing meshes, the cotangent
Laplacian, edges and adjacency, face and vertex normals, and the closest
point on a mesh. Each one follows the libigl function of the same name.
"""
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


# ---------------------------------------------------------------- input / output
def read_triangle_mesh(path):
    """Vertices (n, 3) float and triangles (m, 3) int of an .off, .obj or
    .ply file (ASCII or binary PLY). Polygons are split into fans."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == '.off':
        return _read_off(path)
    if ext == '.obj':
        return _read_obj(path)
    if ext == '.ply':
        return _read_ply(path)
    raise ValueError(f'unsupported mesh format {ext}')


def _fan(polys):
    return np.array([(p[0], p[i], p[i + 1]) for p in polys for i in range(1, len(p) - 1)], dtype=np.int64).reshape(-1, 3)


def _read_off(path):
    tokens = []
    for line in open(path):
        line = line.split('#')[0].strip()
        if line:
            tokens.extend(line.split())
    if tokens[0].upper().endswith('OFF'):
        tokens = [tokens[0][:-3]] + tokens[1:] if len(tokens[0]) > 3 else tokens[1:]
        tokens = [t for t in tokens if t]
    nv, nf = int(tokens[0]), int(tokens[1])
    pos = 3
    vert = np.array(tokens[pos:pos + 3 * nv], float).reshape(nv, 3)
    pos += 3 * nv
    polys = []
    for _ in range(nf):
        k = int(tokens[pos])
        polys.append([int(t) for t in tokens[pos + 1:pos + 1 + k]])
        pos += 1 + k
    return vert, _fan(polys)


def _read_obj(path):
    vert, polys = [], []
    for line in open(path):
        if line.startswith('v '):
            vert.append([float(t) for t in line.split()[1:4]])
        elif line.startswith('f '):
            idx = [int(t.split('/')[0]) for t in line.split()[1:]]
            polys.append([i - 1 if i > 0 else len(vert) + i for i in idx])
    return np.array(vert, float).reshape(-1, 3), _fan(polys)


_PLY_TYPES = {'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1', 'short': 'i2', 'int16': 'i2',
              'ushort': 'u2', 'uint16': 'u2', 'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4',
              'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8'}


def _read_ply(path):
    with open(path, 'rb') as f:
        fmt, elements = None, []
        while True:
            words = f.readline().decode('ascii', 'replace').split()
            if not words:
                continue
            if words[0] == 'format':
                fmt = words[1]
            elif words[0] == 'element':
                elements.append((words[1], int(words[2]), []))
            elif words[0] == 'property':
                elements[-1][2].append(words[1:])
            elif words[0] == 'end_header':
                break
        body = f.read()
    if fmt == 'ascii':
        tokens = body.split()
        pos, read = 0, None

        def read(t, n=1):
            nonlocal pos
            out = np.array(tokens[pos:pos + n], dtype=_PLY_TYPES[t])
            pos += n
            return out
    else:
        end = '<' if fmt == 'binary_little_endian' else '>'
        pos = 0

        def read(t, n=1):
            nonlocal pos
            dt = np.dtype(end + _PLY_TYPES[t])
            out = np.frombuffer(body, dt, n, pos)
            pos += n * dt.itemsize
            return out
    vert, polys = None, []
    for name, count, props in elements:
        if name == 'vertex':
            cols = [p[-1] for p in props]
            if fmt != 'ascii' and all(p[0] != 'list' for p in props):
                dt = np.dtype([(p[1], end + _PLY_TYPES[p[0]]) for p in props])
                rec = np.frombuffer(body, dt, count, pos)
                pos += count * dt.itemsize
                vert = np.column_stack([rec[c] for c in ('x', 'y', 'z')]).astype(float)
                continue
            rows = []
            for _ in range(count):
                rows.append([read(p[-2], int(read(p[1])[0]))
                             if p[0] == 'list' else read(p[0])[0] for p in props])
            vert = np.array([[r[cols.index(c)] for c in ('x', 'y', 'z')] for r in rows], float)
        else:
            for _ in range(count):
                for p in props:
                    if p[0] == 'list':
                        v = read(p[2], int(read(p[1])[0]))
                        if name == 'face' and p[3] in ('vertex_indices', 'vertex_index'):
                            polys.append([int(i) for i in v])
                    else:
                        read(p[0])
    return vert, _fan(polys)


def write_obj(path, vert, triv):
    with open(path, 'w') as f:
        np.savetxt(f, vert, fmt='v %.8f %.8f %.8f')
        np.savetxt(f, np.asarray(triv) + 1, fmt='f %d %d %d')


# ---------------------------------------------------------------- operators
def cotmatrix(vert, triv):
    """Cotangent Laplacian (n, n): L[i, j] = (cot a + cot b) / 2 on the edges,
    negative diagonal (negative semi-definite, as libigl's cotmatrix)."""
    n = len(vert)
    rows, cols, vals = [], [], []
    for k in range(3):
        i, j, o = triv[:, (k + 1) % 3], triv[:, (k + 2) % 3], triv[:, k]
        a, b = vert[i] - vert[o], vert[j] - vert[o]
        cot = np.sum(a * b, axis=1) / np.linalg.norm(np.cross(a, b), axis=1)
        rows += [i, j]
        cols += [j, i]
        vals += [cot / 2, cot / 2]
    L = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return (L - sp.diags(np.asarray(L.sum(axis=1)).ravel())).tocsr()


def edges(triv):
    """Unique undirected edges (e, 2), smaller index first, sorted."""
    e = np.sort(np.concatenate([triv[:, [0, 1]], triv[:, [1, 2]], triv[:, [2, 0]]]), axis=1)
    return np.unique(e, axis=0)


def adjacency_matrix(triv, n=None):
    """Symmetric 0/1 vertex adjacency (n, n) of the mesh edges."""
    e = edges(triv)
    n = int(triv.max()) + 1 if n is None else n
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return (A + A.T).tocsr()


def per_face_normals(vert, triv):
    """Unit normals (m, 3) of the triangles; zero for degenerate ones."""
    n = np.cross(vert[triv[:, 1]] - vert[triv[:, 0]], vert[triv[:, 2]] - vert[triv[:, 0]])
    nrm = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, nrm, out=np.zeros_like(n), where=nrm > 0)


def per_vertex_normals(vert, triv):
    """Unit vertex normals (n, 3): face normals weighted by the face areas
    (libigl's default weighting)."""
    fn = np.cross(vert[triv[:, 1]] - vert[triv[:, 0]], vert[triv[:, 2]] - vert[triv[:, 0]])
    vn = np.zeros_like(vert, dtype=float)
    for k in range(3):
        np.add.at(vn, triv[:, k], fn)
    nrm = np.linalg.norm(vn, axis=1, keepdims=True)
    return np.divide(vn, nrm, out=np.zeros_like(vn), where=nrm > 0)


# ---------------------------------------------------------------- closest point
def _closest_on_triangles(p, a, b, c):
    """Closest point to p on the triangles (a, b, c), all (k, 3)
    (Ericson, Real-Time Collision Detection, 5.1.5)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.sum(ab * ap, 1), np.sum(ac * ap, 1)
    bp = p - b
    d3, d4 = np.sum(ab * bp, 1), np.sum(ac * bp, 1)
    cp = p - c
    d5, d6 = np.sum(ab * cp, 1), np.sum(ac * cp, 1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide='ignore', invalid='ignore'):
        den = va + vb + vc
        v_in, w_in = vb / den, vc / den
        v_ab = d1 / (d1 - d3)
        w_ac = d2 / (d2 - d6)
        w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    out = a + v_in[:, None] * ab + w_in[:, None] * ac                     # inside the face
    cases = [
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + v_ab[:, None] * ab),                   # edge ab
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + w_ac[:, None] * ac),                   # edge ac
        ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), b + w_bc[:, None] * (c - b)),    # edge bc
        ((d6 >= 0) & (d5 <= d6), c),                                                    # vertex c
        ((d3 >= 0) & (d4 <= d3), b),                                                    # vertex b
        ((d1 <= 0) & (d2 <= 0), a),                                                     # vertex a
    ]
    for m, q in cases:   # later cases take precedence, as the early returns of the reference
        out[m] = q[m]
    return out


def point_mesh_squared_distance(points, vert, triv, k=8):
    """For every point: squared distance to the mesh, index of the closest
    triangle and closest point, as libigl's point_mesh_squared_distance.

    Exact. The triangles are grouped by their size (largest distance from
    the centroid to a corner, by powers of 2). In every group the k
    triangles with the nearest centroids are checked, and a point is settled
    once its best distance is at most the distance to the k-th centroid
    minus the largest size in the group, since no unchecked triangle of the
    group can be closer; k doubles for the other points. Among triangles at
    the same distance (a closest point on an edge or a vertex) the first one
    found is returned, which may differ from libigl's choice."""
    points = np.asarray(points, float)
    vert = np.asarray(vert, float)
    triv = np.asarray(triv)
    tri = vert[triv]
    cen = tri.mean(axis=1)
    size = np.linalg.norm(tri - cen[:, None], axis=2).max(axis=1)
    level = np.floor(np.log2(np.maximum(size, size.max() * 1e-6))).astype(int)
    n = len(points)
    sq = np.full(n, np.inf)
    face = np.zeros(n, dtype=np.int64)
    closest = np.zeros_like(points)
    for lv in np.unique(level):
        group = np.flatnonzero(level == lv)
        tree, rad = cKDTree(cen[group]), size[group].max()
        # the points whose best distance already beats every triangle of the group skip it
        todo = np.flatnonzero(tree.query(points)[0] - rad < np.sqrt(sq))
        kk = min(k, len(group))
        while len(todo):
            dc, gi = tree.query(points[todo], kk)
            dc, gi = dc.reshape(len(todo), -1), gi.reshape(len(todo), -1)
            fi = group[gi]
            p = np.repeat(points[todo], fi.shape[1], axis=0)
            f = fi.ravel()
            q = _closest_on_triangles(p, tri[f, 0], tri[f, 1], tri[f, 2])
            d = np.sum((q - p) ** 2, axis=1).reshape(fi.shape)
            j = np.argmin(d, axis=1)
            r = np.arange(len(todo))
            best = d[r, j]
            better = best < sq[todo]
            sq[todo[better]] = best[better]
            face[todo[better]] = fi[r, j][better]
            closest[todo[better]] = q.reshape(len(todo), -1, 3)[r, j][better]
            if kk >= len(group):
                break
            settled = np.sqrt(sq[todo]) <= dc[:, -1] - rad
            todo, kk = todo[~settled], min(2 * kk, len(group))
    return sq, face, closest
