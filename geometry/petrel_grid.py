import numpy as np

from geometry.edge_utils import *
from geometry.petrel_mesh import mesh

# 2--3
# |  |
# 0--1


# def _read_file_argument(grdecl):
#     kwargs = {}
#     dirname = os.path.dirname(grdecl)
#     basename = os.path.basename(grdecl)
#     newfile = "{0}_PYTHON".format(grdecl)
#     if not os.path.isfile(newfile):
#         with open(grdecl) as f:
#             for line in f:
#                 if "*" in line:
#                     rewrite_file(grdecl, newfile)
#                     grdecl = newfile
#                     break
#     else:
#         grdecl = newfile

#     def generator_include(f):
#         for line in f:
#             if "INCLUDE" in line:
#                 yield re.findall("'(.*)'", next(f))[0]

#     with open(grdecl) as f:
#         names = list(generator_include(f))
#     for name in names:
#         oldfile = "{0}/{1}".format(dirname, name)
#         newfile = "{0}/{1}_PYTHON".format(dirname, name)
#         with open(oldfile) as f:
#             for line in f:
#                 if "Generated : Petrel" in line:
#                     break
#             name_variable = line.split()[0] # TODO Comment ça peut marcher l'appel à line en dehors du code ?!
#             if os.path.isfile(newfile):
#                 kwargs[name_variable] = newfile
#             if not os.path.isfile(newfile):
#                 for line in f:
#                     if "*" in line:
#                         rewrite_file(oldfile, newfile)
#                         kwargs[name_variable] = newfile
#                         break
#                 else:
#                     kwargs[name_variable] = oldfile
#     return grdecl, kwargs


# def _rewrite_file(fichier, fichier_out):
#     fout = open(fichier_out, "w")
#     with open(fichier) as f:
#         for line in f:
#             if "*" in line:
#                 data = []
#                 line = line.split()
#                 for value in line:
#                     if value == "/":
#                         data.append(value)
#                     elif "*" in value:
#                         nbr, value = value.split("*") # TODO Ca va planter sur le premier ficher d'exemple
#                         value = float(value)
#                         nbr = int(nbr)
#                         data.extend(nbr * [value])
#                     else:
#                         data.append(float(value))
#                 fout.write("{0}\n".format(" ".join(list(map(str, data)))))
#             elif line.strip() and not line.startswith("--"):
#                 fout.write(line)
#     fout.close()


def pillar_edges(corner_ids, i, j, nz):
    edges = []
    for k in range(nz):
        for l in range(4):
            edge = Edge(corner_ids[i, j, k, l], corner_ids[i, j, k + 1, l + 4])
            if -1 not in edge and edge not in edges:
                edges.append(edge)
    return edges


def map_pillar_edges(edges, nodes):
    new_edges = {}
    for edge in edges:
        noeuds, noeuds_new = [], []
        for node in nodes:
            if noeuds and len(noeuds) < 2:
                noeuds_new.append(node)
            if node in edge:
                if not noeuds:
                    noeuds_new.append(node)
                noeuds.append(node)
        assert len(noeuds) == 2
        assert Edge(noeuds[0], noeuds[1]) == edge
        assert Edge(noeuds_new[0], noeuds_new[-1]) == edge
        assert edge not in new_edges
        if len(noeuds_new) > 2:
            new_edges[edge] = [
                Edge(noeuds_new[u], noeuds_new[u + 1])
                for u in range(len(noeuds_new) - 1)
            ]
    return new_edges


def update_faces_nodes(faces_nodes, faces_edges, all_edges, map_edges):
    """
    For each face, computes its updated contour (ordered list of vertices) by
    taking into account edges that need to be splitted.

    Inputs:
        * faces_nodes: list: for each face, its (ordered) list of nodes
        * faces_edges: list: for each face: its (ordered) list of edges (does NOT
                       directly stores edges, but their indices in all_edges)
        * all_edges:   unordered list: of all the edges (2-tuple of nodes)
                       existing in the faces
        * map_edges:   dictionary containing all (keys =) edges that must be splitted
                       into a (values =) sequence of other existing edges
    Output:
        * faces_nodes: list: for each face, its (ordered) list of nodes,
                       updated (as compared to faces_nodes) by inserting the
                       nodes introduced when splitting edges 
    """
    assert len(faces_nodes) == len(faces_edges), "Inconsistent number of faces"
    for f_id, f_edges_ids in enumerate(faces_edges):
        f_edges = [all_edges[e_id] for e_id in f_edges_ids]
        f_nodes = faces_nodes[f_id]
        for e in f_edges:
            new_edges = map_edges.get(e)
            if new_edges is not None:
                f_nodes = replace_edge(f_nodes, e, new_edges)
        faces_nodes[f_id] = f_nodes
    return faces_nodes


def update_faces_edges(faces_nodes):
    """
    For each face, computes its list of edges from its list of nodes.

    Input:
        * faces_nodes: list: for each face, its (ordered) list of nodes

    Outputs:
        * all_edges:   unordered list: of all the edges (2-tuple of nodes)
                       existing in the faces
        * faces_edges: list: for each face: its (ordered) list of edges (does NOT
                       directly stores edges, but their indices in all_edges)
                       
    """
    faces_edges, all_edges = [], []
    for f_nodes in faces_nodes:
        f_edges = []
        for i in range(len(f_nodes)):
            e = Edge(f_nodes[i - 1], f_nodes[i])
            if e not in all_edges:
                e_id = len(all_edges)
                all_edges.append(e)
            else:
                e_id = all_edges.index(e)
            f_edges.append(e_id)
        faces_edges.append(f_edges)
    return faces_edges, all_edges


def set_faces_nodes(cells):
    # cells décrit les cellules dans l'ordre
    #  vtk :
    #  1-2
    #  | |
    #  4-3
    faces = []
    indices = [
        [0, 1, 2, 3],  # Face haut
        [4, 5, 6, 7],  # Face bas
        [0, 4, 5, 1],  # Face arrière (en Y)
        [3, 7, 6, 2],  # Face avant (en Y)
        [0, 4, 7, 3],  # Face gauche (en X)
        [1, 5, 6, 2],
    ]  # Face droite (en X)
    for i, indice in enumerate(indices):
        faces.extend(np.array(cells)[:, indice].tolist())
    faces_sort = np.sort(faces, axis=1)
    # On élimine les doublons
    newfaces, unique_indices, face_id = np.unique( # FIXME Unused newfaces?
        faces_sort, axis=0, return_index=True, return_inverse=True
    )
    cells_faces = face_id.reshape((len(cells), len(indices)), order="F")
    faces_nodes = np.array(faces)[unique_indices]
    return faces_nodes, cells_faces


def detect_fault(segs1, segs2, pvertices):
    mask = (segs1 != -1) & (segs2 != -1)
    diff = pvertices[segs2][..., 2] - pvertices[segs1][..., 2] # Compare Z of 
    diff = diff[mask]
    if np.nonzero(diff)[0].shape[0] != 0: # nonzero returns a tuple with 1 elt per array dim
        return True                       # we only need to check one dim, to check for nonzero


def build_segs(ix, iy, corner_ids):
    """
    Idea:
     * Definitions:
       * north/south = Y axis, north direction = increasing Y values, south direction = 
                       decreasing Y values
       * east/west = X axis, east direction = increasing X values, west direction = 
                       decreasing X values
       * A pillar = a vertical column of hexahedra in the model
                    pillar(ix, iy) = pillar defined by cells (ix, iy, :)
                    a pillar have 4 vertical faces: notrh, south, east and west
     * Algorithm: 
       * For a given pillar(ix, iy), we compare:
         - the horizontal edges of its north face with the horizontal edges of the
           south face of the pillar(ix, iy + 1) to detect faults along Y
         - the horizontal edges of its east face with the horizontal edges of the
           west face of the pillar(ix + 1, iy) to detect faults along X
       * Note: the "fault detection" step is done later in `detect_fault(...)`. Here
         we only compute the sequences of edges that will be sent to  `detect_fault(...)`
       * Example of the Y face:
         * segs1 = [upper_north_edge for each hexahedron in pillar(ix, iy)]
         * segs2 = [lower_north_edge for each hexahedron in pillar(ix, iy)]
         * segsY1 = segs1.append(segs2[-1]). # that's the "general idea", it's a bit
           smarter in practice =)
         * segs3 = [upper_south_edge for each hexahedron in pillar(ix, iy + 1)]
         * segs4 = [lower_south_edge for each hexahedron in pillar(ix, iy + 1)]
         * segsY2 = segs3.append(segs4[-1]). # that's the "general idea", it's a bit
           smarter in practice =)
         * in `detect_fault(...)`, we compare segsY1 and segsY2
       * For the X face, it's the same with pillar(ix, iy) and pillar(ix + 1, iy),
         and east/west faces.
       * Final note: in practice, indices might be different from what is presented
         here (i.e., I might inverted north/south directions, but you still have the
         general idea of the algorithm !)
    """
    segs1 = np.stack( # [Column (X+1, Y), all Z, V1], [Column (X+1, Y+1), all Z, V3]
        [corner_ids[ix + 1, iy, :, 1], corner_ids[ix + 1, iy + 1, :, 3]], axis=-1
    )
    segs2 = np.stack( # [Column (X+1, Y), all Z, V5], [Column (X+1, Y+1), all Z, V7]
        [corner_ids[ix + 1, iy, :, 5], corner_ids[ix + 1, iy + 1, :, 7]], axis=-1
    )
    segsY1 = np.where(segs1 == -1, segs2, segs1) # [segs1 or segs2 if segs1 is masked]

    segs3 = np.stack(
        [corner_ids[ix + 1, iy, :, 0], corner_ids[ix + 1, iy + 1, :, 2]], axis=-1
    )
    segs4 = np.stack(
        [corner_ids[ix + 1, iy, :, 4], corner_ids[ix + 1, iy + 1, :, 6]], axis=-1
    )
    segsY2 = np.where(segs3 == -1, segs4, segs3)

    segs1 = np.stack(
        [corner_ids[ix, iy + 1, :, 2], corner_ids[ix + 1, iy + 1, :, 3]], axis=-1
    )
    segs2 = np.stack(
        [corner_ids[ix, iy + 1, :, 6], corner_ids[ix + 1, iy + 1, :, 7]], axis=-1
    )
    segsX1 = np.where(segs1 == -1, segs2, segs1)
    segs3 = np.stack(
        [corner_ids[ix, iy + 1, :, 0], corner_ids[ix + 1, iy + 1, :, 1]], axis=-1
    )
    segs4 = np.stack(
        [corner_ids[ix, iy + 1, :, 4], corner_ids[ix + 1, iy + 1, :, 5]], axis=-1
    )
    segsX2 = np.where(segs3 == -1, segs4, segs3)
    return segsY1, segsY2, segsX1, segsX2


def build_segs_pils(grid, segs1, segs2, i, j, u, v, pvertices):
    segs1 = segs1[segs1 != -1].reshape(-1, 2) # "segsY1"
    segs2 = segs2[segs2 != -1].reshape(-1, 2) # "segsY2"
    segs_glo = np.vstack([segs1, segs2]) # segs1.append_elements(segs2)
    nodes1 = np.unique(pillar_edges(grid.corner_ids, i + u, j + v, grid.nz))
    asort = np.argsort(pvertices[nodes1][..., 2])
    nodes1 = nodes1[asort] # All nodes indices on "face 1" ?
    nodes2 = np.unique(pillar_edges(grid.corner_ids, i + 1, j + 1, grid.nz))
    asort = np.argsort(pvertices[nodes2][..., 2])
    nodes2 = nodes2[asort] # All nodes indices on "face 2" ?
    return segs_glo, nodes1, nodes2


def pillar_referential_tri(pts):
    z = pts[:, 2]
    kmin = np.argmin(z)
    kmax = np.argmax(z)
    e = pts[kmax] - pts[kmin]
    return pts[kmin], e


def new_coord_tri(pillarpts, O, e):
    """
    Used to compute the projection of pillar points onto a 2D space.
    Note: here, we only compute the 2nd coordinate (the first one is either 0 or 1,
    see triangulate()) 
    """
    res = np.sum((pillarpts - O) * e, axis=1) # Projection onto a 2D space
    e2 = np.sum(e * e) # Square norm of e
    assert e2 > 0
    res /= e2
    return res


def map_edges_tri(segs_glo, nodes1, nodes2, pvertices):
    asort1 = np.argsort(pvertices[ segs_glo[:, 0] ][:, 2]) # "Sorted" Z of all 1st edge vertex in segs_glo
    asort2 = np.argsort(pvertices[ segs_glo[:, 1] ][:, 2]) # "Sorted" Z of all 2nd edge vertex in segs_glo
    no = segs_glo[:, 0][asort1] # 1st edges vertices sorted by depth
    edges = [Edge(no[u], no[u + 1]) for u in range(len(no) - 1) if no[u] != no[u + 1]]
    # edges = list of vertical edges, sorted according to Z
    map_edges = map_pillar_edges(edges, nodes1)
    no = segs_glo[:, 1][asort2] # 2nd edges vertices sorted by depth
    edges = [Edge(no[u], no[u + 1]) for u in range(len(no) - 1) if no[u] != no[u + 1]]
    map_edges.update(map_pillar_edges(edges, nodes2))
    return map_edges


def build_segs_tri(segs_glo, pvertices):
    ss = segs_glo.flatten(order="F")
    _, indices = np.unique(ss, return_index=True)
    ss = ss[indices]
    vertices_tri = pvertices[ss]
    nodes_tri = np.arange(len(vertices_tri))
    nmap = {ss[i]: nodes_tri[i] for i in range(len(ss))}
    segs_tri = np.array([nmap[s] for s in segs_glo.flatten(order="F")])
    segs_tri = segs_tri.reshape((-1, 2), order="F")
    segs_tri = np.array(segs_tri, dtype=np.int32)
    nmap = {nodes_tri[i]: ss[i] for i in range(len(ss))}
    return nmap, segs_tri, vertices_tri


def map_faces_tri(faces, nmap, pvertices, uv2):
    """Update the indexation of vertices and faces after triangulation.
    In particular:
     * for vertices that already exist in the hexahedric mesh: converts their local
       2D index back to their global 3D index
     * for vertices that were created by the triangulation: creates a new index in
       the 3D global indexation and associates it to the vertex
     * for each face: the face initially stores local 2D indices. These indices are
       converted back to refer to their associated 3D global indices
     * the array of vertices `pvertices` is updated by appending all the vertices
       created by the triangulation to it
    """
    list_vert = pvertices.tolist()
    updated_faces = []
    for face in faces:
        updated_face = []
        for node in face:
            if node in nmap.keys(): # Si node existe dans le modèle hexa (ou a déjà été ajouté)
                updated_face.append(nmap[node])
            else: # Si node a été créé durant la triangulation et n'a pas encore été ajouté
                new_index = len(list_vert)
                list_vert.append(uv2[node])
                updated_face.append(new_index)
                nmap[node] = new_index
        updated_faces.append(updated_face)
    return updated_faces, np.array(list_vert)


def update_faces_tri(faces, map_edges):
    faces2 = []
    for face in faces:
        edges = []
        old_face = face
        for inode in range(len(old_face)):
            edge = Edge(old_face[inode - 1], old_face[inode])
            if edge in map_edges:
                face = replace_edge(face, edge, map_edges[edge])
        faces2.append(face)
    return faces2


def triangulate(segs_glo, nodes1, nodes2, pvertices):
    # Step 1: Pojection of 3D points onto a 2D space (the "fault plane")
    map_edges = map_edges_tri(segs_glo, nodes1, nodes2, pvertices)
    nmap, segs_tri, vertices_tri = build_segs_tri(segs_glo, pvertices)
    _, indices = np.unique(segs_tri[:, 0], return_index=True)
    tri1 = segs_tri[:, 0][indices] # 1st v of each e?
    _, indices = np.unique(segs_tri[:, 1], return_index=True)
    tri2 = segs_tri[:, 1][indices] # 2nd v of each e?
    vertices1 = vertices_tri[tri1]
    vertices2 = vertices_tri[tri2]
    Oback, uback = pillar_referential_tri(vertices1)
    Ofront, ufront = pillar_referential_tri(vertices2)
    nv1 = np.zeros((vertices1.shape[0], 2)) # Projected 2D points
    nv2 = np.ones((vertices2.shape[0], 2)) # Projected 2D points
    nv1[:, 1] = new_coord_tri(vertices1, Oback, uback)
    nv2[:, 1] = new_coord_tri(vertices2, Ofront, ufront)
    nv = np.vstack([nv1, nv2]) # Kind of concatenate: nv = [nv1[0], ..., nv1[m], nv2[0], ..., nv2[n]]
    # Step 2: triangulation of the 2D space
    uv, triangles, components, faces = mesh(nv.astype("float64"), segs_tri)
    # Step 3: Project points back to 3D
    uv2 = np.reshape(uv[:, 0], (-1, 1)) * (
        Ofront + np.tensordot(uv[:, 1], ufront, axes=0)
    )
    uv2 += np.reshape(1 - uv[:, 0], (-1, 1)) * (
        Oback + np.tensordot(uv[:, 1], uback, axes=0)
    )
    # NOTE: faces = list of vertices defining the border of each individual
    #               component after triangulation
    faces, pvertices = map_faces_tri(faces, nmap, pvertices, uv2)
    faces = update_faces_tri(faces, map_edges)
    # Test si on retombe bien sur les précédents points
    assert (
        np.linalg.norm(
            uv2[: nv.shape[0]] - np.vstack([vertices1, vertices2]), axis=1
        ).max()
        < 1e-5
    )
    return nv, uv2, pvertices, components, triangles, faces


def get_cells(uv, triangles, components, segs_glo, pvertices, numcells):
    # centres = For each triangle, it 2D barycenter (in the uv coordinate system)
    centres = np.vstack([uv[triangle].mean(axis=0) for triangle in triangles])
    cells_num, cells_comp = [], []
    segs_glo = segs_glo[segs_glo != -1].reshape(-1, 2) # FIXME Does not necessarily change anything...
    v_sources = pvertices[segs_glo][:, 0] # 1st vertex of each edge in segs_glo
    v_targets = pvertices[segs_glo][:, 1] # 2nd vertex of each edge in segs_glo
    # Computes (a, b) such that: y = ax + b for each edge
    # (note: y = Z axis, x = either X or Y axis, depending on the fault orientations)
    # a = [(t[1] - s[1]) / (t[0] - s[0]) for s, t in zip(v_sources, v_targets)]
    a = (v_targets[:, 1] - v_sources[:, 1]) / (v_targets[:, 0] - v_sources[:, 0])
    # b = [t[1] - a * t[0] for s, t, a in zip(v_targets, a)]
    b = v_targets[:, 1] - a * v_targets[:, 0]
    for component_id, center in zip(components, centres):
        points = a * center[0] + b - center[1]
        for fi, point in enumerate(points):
            if point > 0:
                if fi != 0:
                    cells_num.append(numcells[fi - 1])
                    cells_comp.append(component_id)
                break
    # for i in np.array(cells_num), np.array(cells_comp): # FIXME
    #     print(i)
    return np.array(cells_num), np.array(cells_comp)


# FIXME self as a random parameter name?
# FIXME Unused parameters: num_cell, seg_glo, maxnode
def do_map_edges(self, faces, num_cell, segs_glo, num_old_face, map_edges):
    maxnode = len(self.pvertices)
    edges_faces = Edge_faces(faces)
    keep_edges = []
    for key, value in edges_faces.edge_faces.items():
        if len(value) == 1:
            keep_edges.append(key)
    nodes = [keep_edges[0][0], keep_edges[0][1]]
    edge_done = [keep_edges[0]]
    while len(edge_done) != len(keep_edges):
        for edge in keep_edges:
            if edge not in edge_done:
                v1, v2 = edge
                if v1 == nodes[-1]:
                    nodes.append(v2)
                    edge_done.append(edge)
                    break
                elif v2 == nodes[-1]:
                    nodes.append(v1)
                    edge_done.append(edge)
                    break
    nodes = nodes[:-1] + nodes
    old_nodes = self.faces_nodes[num_old_face]
    for old_node in old_nodes:
        selec_nodes = []
        for node in nodes:
            if node == old_node:
                selec_nodes.append(node)
            elif selec_nodes:
                selec_nodes.append(node)
                if node in old_nodes:
                    break
        if len(selec_nodes) > 2:
            map_edges[Edge(selec_nodes[0], selec_nodes[-1])] = [
                Edge(node, selec_nodes[inode + 1])
                for inode, node in enumerate(selec_nodes[:-1])
            ]
    return map_edges


def locate_face_in_cell(face, cell_faces):
    for i in range(len(cell_faces)):
        if face == cell_faces[i]:
            return i
    # FIXME Unknown variables ?!
    raise ValueError("edge {} not in face {}".format(str(edge), str(face_nodes)))


def replace_face(cell_faces, old_face, new_faces):
    i = locate_face_in_cell(old_face, cell_faces)
    new_cell = [cell_faces[k] for k in range(i + 1, len(cell_faces) + min(0, i - 1))]
    for k in range(max(0, i - 1)):
        new_cell.append(cell_faces[k])
    nk = cell_faces[i - 1]
    new_cell.append(nk)
    for new_face in new_faces:
        new_cell.append(new_face)
        nk = new_cell[-1]
    return new_cell


def get_duplicate_face_id(faces_nodes):
    """
    # FIXME Peut probablement être mieux réécrit"""
    faces_sorted_nodes = [set(f_nodes) for f_nodes in faces_nodes]
    faces_done, new_ids = [], []
    for f_id, f in enumerate(faces_sorted_nodes):
        if f in faces_done:
            id = faces_done.index(f)
            new_ids.append(id)
        else:
            new_ids.append(f_id)
        faces_done.append(f)
    old_ids = np.arange(len(faces_sorted_nodes))
    nmap = {
        old_id: [new_id] for old_id, new_id in zip(old_ids, new_ids) if old_id != new_id
    }
    return nmap


def remove_old_faces(faces_nodes, map_faces):
    """
    Inputs:
        * faces_nodes: just to get the initial number of faces
        * map_faces: # TODO
    """
    old_faces = list(map_faces.keys())
    mask = np.ones(len(faces_nodes), dtype=np.bool)
    mask[old_faces] = False
    return -1 + np.cumsum(mask)


def map_old_new_faces_and_edges(
    self,
    cells_num,
    cells_comp,
    cells_faces,
    faces_tri,
    new_faces_nodes,
    num_face,
    segs_glo,
    map_faces,
    map_edges,
):
    for c_num in np.unique(cells_num):
        c_faces = cells_faces[c_num]  # Numéro de face
        num_old_face = c_faces[num_face]
        c_comp = np.unique(cells_comp[cells_num == c_num])
        faces = [faces_tri[i] for i in c_comp]
        maxnode = len(new_faces_nodes)
        map_faces[num_old_face] = [i for i in range(maxnode, maxnode + len(faces))]
        new_faces_nodes.extend(faces)
        map_edges = do_map_edges(
            self, faces, c_num, segs_glo, num_old_face, map_edges
        )
    return new_faces_nodes, map_faces, map_edges


def replace_old_new_faces(cells_faces, map_faces):
    """
    For each cell, computes its updated list of face ids by taking into account
    the mess done in solve_faults() about { duplicated / splitted / ? } faces.
    Probably (though unchecked): replaces one Quad face by 2 Triangle faces.

    WARNING: does not "create" new cells yet: it just replaces the existing ones.
    So you end up with cells that can have 7 or 8 faces instead of hexahedra, but
    no tetrahedra, pyramids and wedges...

    Inputs:
        * cells_faces: list: for each cell (hexahedron), its list of 6 faces
        * map_faces:   dictionary contaning all (key = ) faces (ids) that must be
                       replace by a (value =) list of other existing faces (ids)
    Output:
        *
    """
    for c_id, c_faces in enumerate(cells_faces):
        new_cell = c_faces
        for f in c_faces:
            new_faces = map_faces.get(f)
            if new_faces is not None:
                new_cell = replace_face(new_cell, f, new_faces)
        cells_faces[c_id] = new_cell
    return cells_faces


def solve_fault(
    grid, i, j, axe, pvertices, cells_faces, faces_nodes, map_edges, map_faces
):
    segs1, segs2 = axe[0]
    ipil, jpil = axe[1]
    iverts = axe[2]
    iface1, iface2 = axe[3]
    segs_glo, nodes1, nodes2 = build_segs_pils(
        grid, segs1, segs2, i, j, ipil, jpil, pvertices
    )
    # _ = nv = points after projection in 2D
    # uv2 = 3D points (2D points after triangulation that were projected back to 3D)
    # pvertices = all 3D points (updated with vertices that were created by the triangulation)
    # components = for each triangle (after triangulation), the id of the component it belongs to
    # triangles = for each triangle (after triangulation), its 3 vertices
    # faces_tri = for each individual component (after triangulation), the ordered list
    #             of vertices defining its border
    _, uv2, pvertices, components, triangles, faces_tri = triangulate(
        segs_glo, nodes1, nodes2, pvertices
    )
    sides = {1: [i, j, iface1, segs1], 2: [i + ipil, j + jpil, iface2, segs2]}
    # For each side of the fault (the face of pillar1 and the face of pillar2 in contact with the fault)
    # print("In solve_faults: segs_glo = ") # FIXME
    # print(segs_glo)
    for key in sides.keys():
        w = sides[key]
        numcells = grid.numcells[w[0], w[1]]
        # uv2[:, iverts] = 2D version of uv2, with the "uninteresting" dimension removed (X or Y)
        cells_num, cells_comp = get_cells(
            uv2[:, iverts], triangles, components, w[3], pvertices[:, iverts], numcells
        )
        faces_nodes, map_faces, map_edges = map_old_new_faces_and_edges(
            grid,
            cells_num,
            cells_comp,
            cells_faces,
            faces_tri,
            faces_nodes,
            w[2],
            segs_glo,
            map_faces,
            map_edges,
        )
    return pvertices, faces_nodes, map_edges, map_faces


def new_cell_faces(cells_faces, new_ids):
    """For each cell, updates its list of faces using the new faces indexing"""
    return [new_ids[c_faces] for c_faces in cells_faces]


def passe_finale(cells_faces, faces_nodes, map_faces):
    cells_faces = replace_old_new_faces(cells_faces, map_faces)
    new_ids = remove_old_faces(faces_nodes, map_faces)
    cells_faces = new_cell_faces(cells_faces, new_ids)
    new_ids, index = np.unique(new_ids, return_index=True)
    faces_nodes = [faces_nodes[i] for i in index]
    return cells_faces, faces_nodes


class PetrelGrid(object):
    def extract_data_from_files(self, mainfile, **kwargs):
        #  Lecture du nom des différents fichiers
        zcornfile = kwargs.get("ZCORN", mainfile)
        coordfile = kwargs.get("COORD", mainfile)
        actnumfile = kwargs.get("ACTNUM", mainfile)
        permxfile = kwargs.get("PERMX", mainfile)
        permyfile = kwargs.get("PERMY", mainfile)
        permzfile = kwargs.get("PERMZ", mainfile)
        # Lecture des informations de grilles.
        # On lit les coordonnées de type "Corner Point Geometry"
        #  avec les MOTS-CLES : COORD et ZCORN
        # Direction Z vertical du haut vers le bas
        # Direction X horizontal de gauche à droite
        # Direction Y horizontal de l'arrière vers l'avant
        # MOT-CLE COORD : triplet X Y Z X Y Z haut et bas de la grille
        # 1er triplet : coordonnées du haut de la grille
        #  2ème triplet : coordonnées du bas de la grille
        # MOT-CLE ZCOORN : 8 profondeurs définies pour chaque cellule
        #  pour les 8 coins. Les valeurs sont données d'abord pour la face
        #  du haut :
        #  1-2
        #  | |
        #  3-4
        #  Puis pour la face du bas
        #  5-6
        #  | |
        #  7-8
        #  Soit zij la profondeur, N nombre de cellules,
        #  NX dimension X, NY dimension Y, NZ dimension Z
        # i indice des cellules (1 <= i <= N)
        #  j indices des coins (1 <= j <= 8)
        #  Les valeurs sont rangées dans cet ordre :
        #  - Pour toutes les NX cellules du 1er plan et de la première ligne
        #    - D'abord z de la face haute (z11, z12, z21, z22..., zNX1, zNX2)
        #    - Puis (z13, z14, z23, z24..., zNX3, zNX4)
        #   - Répétition pour les NY - 1 lignes de la grille
        #    - Ensuite z de la face basse (z15, z16, z25, z26..., zNX5, zNX6)
        #    - Puis (z17, z18, z27, z28..., zNX7, zNX8)
        #   - Répétition pour les NY - 1 lignes de la grille
        #  - Même chose pour les NZ - 1 plans suivant
        with open(mainfile) as f:
            for line in f:
                # if line.startswith('MAPAXES'):
                # self.mapaxes = np.asarray(next(f).strip().split(' ')[:-1],
                # 'float')
                if line.startswith("SPECGRID"):
                    self.specgrid = self.nx, self.ny, self.nz = np.asarray( # FIXME C'est pas très jojo tout ça...
                        # next(f).strip().split(" ")[:-3], "int" # FIXME Dans mes exemples il n'y a que 2 valeurs à supprimer
                        next(f).strip().split(" ")[:-2], "int"
                    )
        #  Tableau des coordonnées. Shape ((NX+1)*(NY+1), 6)
        coord = self._read_coord(coordfile)
        pillars = coord.reshape((self.nx + 1, self.ny + 1, 6), order="F")
        #  Tableau des zcorners. Shape (NX, NY, NZ, 8)
        self._read_zcorn(zcornfile)
        self.permx = self._read_grid(permxfile, "PERMX")
        self.permy = self._read_grid(permyfile, "PERMY")
        self.permz = self._read_grid(permzfile, "PERMZ")
        actnum = self._read_grid(actnumfile, "ACTNUM")
        return pillars, actnum

    def build_grid(self, pillars, actnum=None):
        # Vérifier que la correspondance des couches est cohérente (pas de trous)
        #  Zcorners des faces hautes correspondent-elles bien aux zcorners des
        #  faces basses ?
        assert np.all(self.zcorn[:, :, 1:, :4] == self.zcorn[:, :, :-1, 4:])
        self.x = np.zeros(self.zcorn.shape)
        self.y = np.zeros(self.zcorn.shape)
        #  Grille des noeuds hauts et bas. Shape ((NX+1), (NY+1), 6)
        #  Epaisseur totale de la grille sur chaque pilier
        #  Bas moins haut pour avoir une épaisseur positive
        dxyz = pillars[:, :, 3:] - pillars[:, :, :3]
        #  Tests si piliers d'épaisseur nul ou positif
        bad_pillars = dxyz[:, :, 2] == 0
        if np.any(bad_pillars):
            print("WARNING You have", np.sum(bad_pillars), "bad pillars!")
        assert np.all(dxyz[:, :, 2] >= 0)
        # Vecteur directeur
        #  0 : face haut. 4 : face basse (indices python)
        for i in [0, 4]:
            for k, pos in enumerate(
                [
                    (slice(None, -1), slice(None, -1)),
                    (slice(1, None), slice(None, -1)),
                    (slice(None, -1), slice(1, None)),
                    (slice(1, None), slice(1, None)),
                ]
            ):
                for izc in range(self.nz):
                    # Vecteur directeur dirigé du haut vers le bas, calculé pour chaque pilier
                    dz = self.zcorn[:, :, izc, i + k] - pillars[pos + (2,)]
                    # Calcul des coordonnées x y de chaque noeud sur chaque pilier
                    self.x[:, :, izc, i + k] = np.where(
                        dxyz[pos + (2,)] == 0,
                        pillars[pos + (0,)],
                        pillars[pos + (0,)] + dz * dxyz[pos + (0,)] / dxyz[pos + (2,)],
                    )
                    self.y[:, :, izc, i + k] = np.where(
                        dxyz[pos + (2,)] == 0,
                        pillars[pos + (1,)],
                        pillars[pos + (1,)] + dz * dxyz[pos + (1,)] / dxyz[pos + (2,)],
                    )
        # Test des correspondances des noeuds en x et y :
        # les points de la face inférieure d'un cube correspondent bien à ceux de la face supérieure du cube en dessous
        assert np.all(self.x[:, :, 1:, :4] == self.x[:, :, :-1, 4:])
        assert np.all(self.y[:, :, 1:, :4] == self.y[:, :, :-1, 4:])
        # On enlève les cellules plates
        #  Masque des cellules plates mask_cells.
        #  mask_cells Shape (NX, NY, NZ, 4)
        mask_cells = np.zeros(self.zcorn.shape[:3] + (4,))
        for i in range(4):
            mask_cells[self.zcorn[:, :, :, i] == self.zcorn[:, :, :, i + 4]] = 1
        #  mask_cells Shape (NX, NY, NZ)
        mask_cells = np.sum(mask_cells, axis=-1)
        # Tests si certaines cellules ont moins de 4 arêtes plates
        for i in range(1, 3):
            assert np.all(mask_cells != 1)
            assert np.all(mask_cells != 2)
            assert np.all(mask_cells != 3)
        mask_cells = np.where(mask_cells == 4, 0, 1)
        # On ajoute les cellules inactives plates au masque
        #  1 : cellule active. 0 : cellule inactive
        if actnum is not None:
            mask_cells = np.where(actnum == 0, 0, mask_cells)
        self.mask_cells = mask_cells
        #  mask_cells Shape (NX, NY, NZ, 8)
        mask_cells = np.repeat(mask_cells[:, :, :, np.newaxis], 8, axis=-1)
        #  9999. valeur des noeuds inactifs
        self.x = np.where(mask_cells == 0, 9999.0, self.x)
        self.y = np.where(mask_cells == 0, 9999.0, self.y)
        self.zcorn = np.where(mask_cells == 0, 9999.0, self.zcorn)
        #  Numérotation des cellules
        #  self.mask_cells Shape (NX, NY, NZ)
        numcells = np.cumsum(self.mask_cells) - 1
        self.numcells = numcells.reshape(self.mask_cells.shape)
        del mask_cells

    @classmethod
    def build_from_files(self, mainfile, **kwargs):
        grid = PetrelGrid()
        grid.build_grid(*grid.extract_data_from_files(mainfile, **kwargs))
        return grid

    @classmethod
    def build_from_arrays__for_dummy_grids(self, pillar_tops, pillar_bottoms, zcorn):
        """ Only used for test dummy_grid from MeshTools """
        # FIXME Fixer les fixme suivants (faire attention, il en existe peut-être plus)
        #       Appel du type : PetrelGrid.build_from_arrays__for_dummy_grids(p1, p2, p3)
        #       => les self.attributs désignent des attributs de classe et non d'instance ! <smiley_qui_vomit> !
        grid = PetrelGrid()
        ncx, ncy, dim = pillar_tops.shape
        assert dim == 3
        nx, ny = ncx - 1, ncy - 1
        self.nx = nx # FIXME c'est quoi ce self ?! Ce devrait être un cls ? A qui appartiennent nx, ny et nz ??? 
        self.ny = ny
        self.nz = zcorn.shape[2]
        assert nx > 0 and ny > 0
        assert pillar_tops.shape == pillar_bottoms.shape
        pillars = np.zeros((ncx, ncy, 6), dtype=np.double)
        for i in range(ncx):
            for j in range(ncy):
                pillars[i, j, :3] = pillar_tops[i, j, :]
                pillars[i, j, 3:] = pillar_bottoms[i, j, :]
        grid.zcorn = zcorn
        grid.build_grid(pillars)
        self.permx = None # FIXME c'est quoi ce self ?! Ce devrait être un cls ? A qui appartiennent permx, permy et permz ??? 
        self.permy = None
        self.permz = None
        return grid

    def _read_grid(self, mainfile, field, **kwargs):
        with open(mainfile) as f:
            line = f.readline()
            while line:
                if line.startswith(field):
                    return np.fromfile(
                        f, sep=" ", count=self.nx * self.ny * self.nz, **kwargs
                    ).reshape((self.nx, self.ny, self.nz), order="F")
                line = f.readline()

    def _read_coord(self, mainfile):
        with open(mainfile) as f:
            line = f.readline()
            while line:
                if line.replace("COORDSYS", "").startswith("COORD"):
                    coord = np.fromfile(
                        f, sep=" ", count=6 * (self.nx + 1) * (self.ny + 1)
                    ).reshape((-1, 6))
                    return coord
                line = f.readline()

    def _read_zcorn(self, mainfile):
        self.zcorn = np.zeros((self.nx, self.ny, self.nz, 8))
        with open(mainfile) as f:
            line = f.readline()
            while line:
                if line.startswith("ZCORN"):
                    for k in range(self.nz):
                        for j in range(self.ny):
                            self.zcorn[:, j, k, 0:2] = np.fromfile(f, sep=" ", count=2 * self.nx).reshape((self.nx, 2))
                            self.zcorn[:, j, k, 2:4] = np.fromfile(f, sep=" ", count=2 * self.nx).reshape((self.nx, 2))
                        for j in range(self.ny):
                            self.zcorn[:, j, k, 4:6] = np.fromfile(f, sep=" ", count=2 * self.nx).reshape((self.nx, 2))
                            self.zcorn[:, j, k, 6:8] = np.fromfile(f, sep=" ", count=2 * self.nx).reshape((self.nx, 2))
                    break
                line = f.readline()

    def process(self):
        """
        Returns:
         * hexahedra: np.array((NbHexa, 8)): for each hexahedron, the 8 UIds of its vertices
         * vertices: np.array((NbVertices, 3)): for each vertex, its 3 (X, Y, Z) coordinates
         * cell_faces: np.array((NbHexa, 6)): for each hexahedron, the 6 UIds of its faces
         * face_nodes: np.array((NbFaces, 4)): for each (unique) face: the 4 UIds of its vertices
        """
        vertices, ids = [], []
        new_ids = np.zeros(8, dtype=np.long)
        # Coordonnées X, Y et Z de chaque pilier. Shape (NX+1,NY+1,NZ+1,8)
        # Va contenir pour chaque noeud de chaque pilier les coordonnées
        # des noeuds de chaque cellule
        xpil = np.ones((self.nx + 1, self.ny + 1, self.nz + 1, 8)) * 9999
        ypil = np.ones((self.nx + 1, self.ny + 1, self.nz + 1, 8)) * 9999
        zpil = np.ones((self.nx + 1, self.ny + 1, self.nz + 1, 8)) * 9999
        #  Attribution à chaque noeud du pilier des valeurs
        for u, v in zip([xpil, ypil, zpil], [self.x, self.y, self.zcorn]):
            #  On attribue les coordonnées des faces hautes à tous les plans
            # des noeuds
            u[:-1, :-1, :-1, 0] = v[:, :, :, 0]
            u[1:, :-1, :-1, 1] = v[:, :, :, 1]
            u[:-1, 1:, :-1, 2] = v[:, :, :, 2]
            u[1:, 1:, :-1, 3] = v[:, :, :, 3]
            #  On fait de même avec les faces basses à tous
            # les plans des noeuds
            u[:-1, :-1, 1:, 4] = v[:, :, :, 4]
            u[1:, :-1, 1:, 5] = v[:, :, :, 5]
            u[:-1, 1:, 1:, 6] = v[:, :, :, 6]
            u[1:, 1:, 1:, 7] = v[:, :, :, 7]
        xpil.shape = (-1, 8)
        ypil.shape = (-1, 8)
        zpil.shape = (-1, 8)
        #  1. Vire les sommets confondus pour ne garder qu'un id par sommet
        # Boucle l'ensemble des noeuds de la grille
        for zci, zc in enumerate(zpil):
            #  Boucle sur les "noeuds" des noeuds de la grille
            for iy in range(len(zc)):  # len(z) = 8
                # Boucle sur les 8 valeurs de zc
                for iz in range(iy):
                    if zc[iz] == zc[iy]:
                        new_ids[iy] = new_ids[iz]
                        break
                #  Détection d'un décalage en z : possible faille
                else:
                    # Si z = 9999., point masqué on ne le rajoute pas
                    if zc[iy] != 9999.0:
                        X = (xpil[zci][iy], ypil[zci][iy], zc[iy])
                        if X not in vertices:
                            new_ids[iy] = len(vertices)
                            vertices.append(X)
                        else:
                            indice = vertices.index(X)
                            new_ids[iy] = indice
            ids.append(np.copy(new_ids))  # la copie est importante ici
        # normalement ici on a le tableaux des vertices et 'yapluka' construire tous les hexagones
        vertices = np.array(vertices)
        corner_ids = np.array(ids)
        #  Ids des noeuds. Shape (NX+1,NY+1,NZ+1,8)
        corner_ids = corner_ids.reshape((self.nx + 1, self.ny + 1, self.nz + 1, 8))
        # Boucle imbriquée
        hexahedra = []
        for ix in range(self.nx):
            for iy in range(self.ny):
                for iz in range(self.nz):
                    if self.mask_cells[ix, iy, iz] == 1:
                        # Attention ! Les noeuds de l'hexaedre sont
                        # définis dans le sens vtk.
                        # C'est à dire :
                        #  1-2
                        #  | |
                        #  4-3
                        hexahedron = (
                            corner_ids[ix, iy, iz, 0],
                            corner_ids[ix + 1, iy, iz, 1],
                            corner_ids[ix + 1, iy + 1, iz, 3],
                            corner_ids[ix, iy + 1, iz, 2],
                            corner_ids[ix, iy, iz + 1, 4],
                            corner_ids[ix + 1, iy, iz + 1, 5],
                            corner_ids[ix + 1, iy + 1, iz + 1, 7],
                            corner_ids[ix, iy + 1, iz + 1, 6],
                        )
                        hexahedra.append(hexahedron)
        self.pvertices = vertices
        zpil = zpil.reshape(corner_ids.shape)
        corner_ids = np.where(zpil == 9999.0, -1, corner_ids)
        self.corner_ids = corner_ids
        faces_nodes, cells_faces = set_faces_nodes(hexahedra)
        self.faces_nodes = faces_nodes
        self.cells_faces = cells_faces
        return np.asarray(hexahedra, dtype=np.int64), vertices, cells_faces, faces_nodes # FIXME dtype OK ?

    def _get_perm(self):
        permx, permy, permz = [], [], []
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    if self.mask_cells[i, j, k] == 1:
                        if self.permx is not None:
                            permx.append(self.permx[i, j, k])
                        if self.permy is not None:
                            permy.append(self.permy[i, j, k])
                        if self.permz is not None:
                            permz.append(self.permz[i, j, k])
        return permx, permy, permz

    def process_faults(self, cells): # FIXME Unused parameter cells
        # Note about comments: c = cell, f = face(t), v = vertex
        new_faces_nodes = self.faces_nodes.tolist()  # List of list, shape = (nb_f, 4), [v_ID] for each v in f
        cells_faces = self.cells_faces.tolist() # List of list, shape = (nb_c, 6), [f_ID] for each f in c
        pvertices = self.pvertices # 2D numpy array, shape = (nb_v, 3), [X, Y, Z] for each v
        map_edges, map_faces = {}, {}
        for ix in range(self.nx): # self.nx = nb c along X
            for iy in range(self.ny): # self.ny = nb c along Y
                # self.corner_ids = np.where(zpil == 9999.0, -1, corner_ids)
                # Shape of self.corner_ids = (self.nx+1, self.ny+1, self.nz+1, 8), 8 = nb_v in c
                segsY1, segsY2, segsX1, segsX2 = build_segs(ix, iy, self.corner_ids)
                axes = {
                    "Y": [(segsY1, segsY2), (1, 0), (1, 2), (5, 4)],
                    "X": [(segsX1, segsX2), (0, 1), (0, 2), (3, 2)],
                }
                for axe in ["Y", "X"]:
                    segs1, segs2 = axes[axe][0]
                    if detect_fault(segs1, segs2, pvertices):
                        (
                            pvertices,
                            new_faces_nodes,
                            map_edges,
                            map_faces,
                        ) = solve_fault(
                            self,
                            ix,
                            iy,
                            axes[axe],
                            pvertices,
                            cells_faces,
                            new_faces_nodes,
                            map_edges,
                            map_faces,
                        )
        # "Mise à jour" du modèle après calcul des nouveaux points, arêtes et faces
        # FIXME Most computation time occurs between this point and the return statement
        faces_edges, edges = update_faces_edges(new_faces_nodes)
        new_faces_nodes = update_faces_nodes(
            new_faces_nodes, faces_edges, edges, map_edges
        )
        cells_faces, new_faces_nodes = passe_finale(
            cells_faces, new_faces_nodes, map_faces
        )
        map_duplicate_faces = get_duplicate_face_id(new_faces_nodes)
        cells_faces, new_faces_nodes = passe_finale(
            cells_faces, new_faces_nodes, map_duplicate_faces
        )
        return pvertices, cells_faces, new_faces_nodes
