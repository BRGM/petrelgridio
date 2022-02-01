from itertools import product
import numpy as np


def petrel_unit_cube(dtype=np.double):
    """Build a unit cube defined by its 8 nodes
    in the order expected by Petrel"""
    nodes = np.zeros((8, 3), dtype=dtype)
    nodes[1::2, 0] = 1  # right face  x = 1
    nodes[2:4 , 1] = 1  # back face   y = 1
    nodes[6:  , 1] = 1  # back face   y = 1
    nodes[4:  , 2] = 1  # bottom face z = 1 (3rd coord is depth!)
    return nodes


def add_buffer(hexaedra):
    ncx, ncy, ncz, _, dim = hexaedra.shape
    assert ncx > 0 and ncy > 0 and ncz > 0 and _ == 8 and dim == 3
    buffered = np.empty((ncx + 2, ncy + 2, ncz, 8, 3), dtype=hexaedra.dtype)
    buffered[1 : (ncx + 1), 1 : (ncy + 1), ...] = hexaedra
    buffered[0, 1 : (ncy + 1), :, 1::2] = hexaedra[0, :, :, 0::2]
    buffered[ncx + 1, 1 : (ncy + 1), :, 0::2] = hexaedra[-1, :, :, 1::2]
    buffered[1 : (ncx + 1), 0, :, (2, 3, 6, 7)] = hexaedra[:, 0, :, (0, 1, 4, 5)]
    buffered[1 : (ncx + 1), ncy + 1, :, (0, 1, 4, 5)] = hexaedra[:, -1, :, (2, 3, 6, 7)]
    buffered[0, 0, :, (3, 7)] = hexaedra[0, 0, :, (0, 4)]
    buffered[0, -1, :, (1, 5)] = hexaedra[0, -1, :, (2, 6)]
    buffered[-1, 0, :, (2, 6)] = hexaedra[-1, 0, :, (1, 5)]
    buffered[-1, -1, :, (0, 4)] = hexaedra[-1, -1, :, (3, 7)]
    return buffered


def add_mask_buffer(hexaedra):
    buffered = np.ma.array(add_buffer(hexaedra))
    mask = np.zeros(buffered.shape, dtype=np.bool)
    mask[0, :, :, (0, 2, 4, 6), :] = True
    mask[-1, :, :, (1, 3, 5, 7), :] = True
    mask[:, 0, :, (0, 1, 4, 5), :] = True
    mask[:, -1, :, (2, 3, 6, 7), :] = True
    buffered.mask = mask
    return buffered


def collect_pillars_nodes(hexaedra):
    ncx, ncy, ncz, _, dim = hexaedra.shape
    assert ncx > 0 and ncy > 0 and ncz > 0 and _ == 8 and dim == 3
    buffered = add_buffer(hexaedra)
    pillars_nodes = np.concatenate(
        (
            buffered[:-1, :-1, :, (3, 7), :],
            buffered[:-1, 1:, :, (1, 5), :],
            buffered[1:, :-1, :, (2, 6), :],
            buffered[1:, 1:, :, (0, 4), :],
        ),
        axis=3,
    )
    assert pillars_nodes.shape[:2] == (ncx + 1, ncy + 1)
    assert np.all(pillars_nodes[..., 1::2, 2] >= pillars_nodes[..., 0::2, 2])
    return pillars_nodes


def pillars(hexaedra):
    pillars_nodes = collect_pillars_nodes(hexaedra)
    i, j = np.indices(pillars_nodes.shape[:2])
    top = pillars_nodes[:, :, 0, 0::2]
    bottom = pillars_nodes[:, :, -1, 1::2]
    pillars = np.concatenate(
        (
            top[i, j, np.argmin(top[..., 2], axis=2), None],  # z is depth
            bottom[i, j, np.argmax(bottom[..., 2], axis=2), None],  # z is depth
        ),
        axis=-1,
    )
    assert pillars.shape[2] == 1
    return pillars[..., 0, :]


def four_cells_stairs(vertical_translation=0.2, cube=None):
    if cube is None:
        cube = petrel_unit_cube()
    t = lambda x, y, z: np.array((x, y, z), dtype=cube.dtype)
    cells = np.array(
        [
            cube + t(i, j, k * vertical_translation)
            for k, (i, j) in enumerate(product((0, 1), (0, 1)))
        ]
    )
    cells.shape = (2, 2, 1, 8, 3)
    return cells


def grid_of_heaxaedra(shape, cube=None):
    assert np.all(np.array(shape) > 0) and len(shape) == 3
    if cube is None:
        cube = petrel_unit_cube()
    t = lambda x, y, z: np.array((x, y, z), dtype=cube.dtype)
    cells = np.array(
        [cube + t(i, j, k) for i, j, k in product(*(range(n) for n in shape))]
    )
    cells.shape = shape + (8, 3)
    return cells


def faulted_ramp(shape, begin=0.5, voffset=1.0):
    nx, ny, nz = shape
    assert nx > 0 and ny > 0 and nz > 0 and ny % 2 == 0
    cells = grid_of_heaxaedra(shape)
    L = float(nx)
    half = cells[:, : (ny // 2), ...]
    x = half[..., 0]
    x0 = begin * L
    offset = np.zeros(half.shape[:-1], dtype=np.double)
    where = x > x0
    offset[where] = voffset * (1 - np.cos((x[where] - x0) * (np.pi / L)))
    cells[:, : (ny // 2), ..., 2] += offset
    return cells


def common_node():
    cube = petrel_unit_cube()
    t = lambda x, y, z: np.array((x, y, z), dtype=cube.dtype)
    cells = np.array([cube, cube + t(0, 0, 1), cube + t(0, 1, -1), cube + t(0, 1, 0),])
    cells.shape = (1, 2, 2, 8, 3)
    cells[0, 1, :, 1::2, 2] -= 0.1  # 1::2 right faces
    return cells


def depth_to_elevation(pts, zref=None, zmap=None, copy=False):
    """Se contente d'inverser le Z des points : Z = -Z"""
    if zmap is not None:
        assert zref is None
        assert False, "not implemented"
    elif zref is None:
        zref = 0
    assert pts.shape[-1] == 3
    res = np.array(pts, copy=copy)
    res[..., 2] = zref - res[..., 2]
    return res


def minimum_distance(pts):
    n, dim = pts.shape
    assert dim == 3
    return np.min(
        [
            np.linalg.norm(pts[j] - pts[i])
            for i, j in product(range(n), range(n))
            if i < j
        ]
    )


def build_regular(nx, nz, xres=1, yres=1, zres=1):
    cube = petrel_unit_cube()
    cube[:, 0] = cube[:, 0] * xres
    cube[:, 1] = cube[:, 1] * yres
    cube[:, 2] = cube[:, 2] * zres
    t = lambda x, y, z: np.array((x, y, z), dtype=cube.dtype)
    mesh = []
    for i in range(nx):
        cols = []
        for j in range(nz):
            cols.append(cube + t(i * xres, 0, j * zres))
        mesh.append(np.array(cols, dtype=cube.dtype))
    mesh = np.array(mesh, dtype=cube.dtype)
    return mesh


def shift_depth(col, h):
    col[:, :, 2] = col[:, :, 2] + h
    col[0, :4, 2] = col[0, :4, 2] - h
    col[-1, 4:, 2] = col[-1, 4:, 2] - h
    return col


def build_non_conformable_mesh(nx, nz, h, **kwargs):
    mesh = build_regular(nx, nz, **kwargs)
    for i in range(1, nx):
        mesh[i, :, :, :] = shift_depth(mesh[i, :, :, :], i * h)
    return mesh


def build_one_unconformity_mesh(nx, nz, h, **kwargs):
    mesh = build_regular(nx, nz, **kwargs)
    for i in range(int(nx / 2), nx):
        mesh[i, :, :, :] = shift_depth(mesh[i, :, :, :], h)
    return mesh


def split_cube(pcube, n):
    cubes = []
    for i in range(n):
        cube = pcube.copy()
        delta = (pcube[4:, 2] - pcube[:4, 2]) / n
        cube[:4, 2] = pcube[:4, 2] + i * delta
        cube[4:, 2] = pcube[:4, 2] + (i + 1) * delta
        cubes.append(cube)
    return np.array(cubes, dtype=cube.dtype)


def refine(cubes, n):
    cubes2 = []
    for i in range(cubes.shape[0]):
        splitted_cubes = []
        for k in range(cubes.shape[1]):
            splitted_cubes.extend(split_cube(cubes[i, k], n))
        splitted_cubes = np.array(splitted_cubes, dtype=cubes.dtype)
        cubes2.append(splitted_cubes)
    return np.array(cubes2, dtype=cubes.dtype)
