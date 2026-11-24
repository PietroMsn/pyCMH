"""cmh.mesh on meshes with known answers (no libigl needed)."""
import numpy as np
import pytest

from cmh import mesh as M

# unit square in the z = 0 plane, two triangles
SQ_V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
SQ_F = np.array([[0, 1, 2], [0, 2, 3]])


def test_cotmatrix_square():
    L = M.cotmatrix(SQ_V, SQ_F).toarray()
    assert np.allclose(L, L.T)
    assert np.allclose(L.sum(axis=1), 0)
    # the diagonal edge 0-2 is opposite two right angles: cot = 0
    assert np.isclose(L[0, 2], 0)
    # edge 0-1 is opposite a 45 degree angle: (cot 45) / 2
    assert np.isclose(L[0, 1], 0.5)


def test_normals_square():
    assert np.allclose(M.per_face_normals(SQ_V, SQ_F), [0, 0, 1])
    assert np.allclose(M.per_vertex_normals(SQ_V, SQ_F), [0, 0, 1])


def test_edges_adjacency():
    assert M.edges(SQ_F).tolist() == [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]]
    A = M.adjacency_matrix(SQ_F).toarray()
    assert A.sum() == 10 and np.allclose(A, A.T)


def test_closest_point_square():
    P = np.array([[0.25, 0.5, 2.0],     # above the inside
                  [2.0, 0.5, 0.0],      # beside an edge
                  [-1.0, -1.0, 1.0]])   # beyond a corner
    sq, face, q = M.point_mesh_squared_distance(P, SQ_V, SQ_F)
    assert np.allclose(q, [[0.25, 0.5, 0], [1, 0.5, 0], [0, 0, 0]])
    assert np.allclose(sq, [4, 1, 3])
    assert face[0] == 1


def test_closest_point_brute_force():
    rng = np.random.default_rng(0)
    V = rng.standard_normal((60, 3))
    F = np.array([rng.choice(60, 3, replace=False) for _ in range(120)])
    F[:5] = [[0, 1, 2]] * 5                      # duplicates and a large triangle
    V[:3] *= 5
    P = 3 * rng.standard_normal((300, 3))
    sq, _, q = M.point_mesh_squared_distance(P, V, F)
    brute = np.min([M.point_mesh_squared_distance(P, V, F[[i]])[0] for i in range(len(F))], axis=0)
    assert np.allclose(sq, brute)
    assert np.allclose(np.sum((q - P) ** 2, axis=1), sq)


@pytest.mark.parametrize('ext', ['off', 'obj'])
def test_read_write(tmp_path, ext):
    path = tmp_path / f'm.{ext}'
    if ext == 'obj':
        M.write_obj(path, SQ_V, SQ_F)
    else:
        path.write_text('OFF\n4 2 0\n' + '\n'.join(' '.join(map(str, v)) for v in SQ_V)
                        + '\n3 0 1 2\n3 0 2 3\n')
    V, F = M.read_triangle_mesh(path)
    assert np.allclose(V, SQ_V) and np.array_equal(F, SQ_F)


def test_read_ply_binary(tmp_path):
    path = tmp_path / 'm.ply'
    header = ('ply\nformat binary_little_endian 1.0\nelement vertex 4\nproperty float x\nproperty float y\n'
              'property float z\nelement face 2\nproperty list uchar int vertex_indices\nend_header\n')
    body = SQ_V.astype('<f4').tobytes() + b''.join(np.uint8(3).tobytes() + f.astype('<i4').tobytes() for f in SQ_F)
    path.write_bytes(header.encode() + body)
    V, F = M.read_triangle_mesh(path)
    assert np.allclose(V, SQ_V) and np.array_equal(F, SQ_F)
