import numpy as np

from pycgal.Epick import Point_2, Point_3
import pycgal.Mesh_2 as Mesh_2
import pycgal.Polygon_mesh_processing as PMP
import pycgal.Surface_mesh as SM


def _build_constrained_delaunay_triangulation(vertices, edges):
    # Creates empty mesh
    cdt = Mesh_2.Constrained_Delaunay_triangulation_2_with_intersections()
    # Inserts all vertices, registers pairs of (vertex id, vertex handle)
    map_vertices = {i: cdt.insert(Point_2(*v)) for i, v in enumerate(vertices)}
    # Inserts all edges as constraints in the triangulation
    for e in edges:
        cdt.insert_constraint(map_vertices[e[0]], map_vertices[e[1]])
    assert cdt.number_of_faces() > 0
    # Note: cdt mesh is updated at each vertex/constraint insertion
    return cdt


def _compute_connected_components(cdt):
    """
    Converts the Mesh_2.Constrained_Delaunay_triangulation to a Surface_mesh,
    and computes the connected components.
    Returns: the newly created mesh, and the property_map storing for each face
    in the model the id of the component it belongs to.
    """
    # Inputs
    vertices, triangles, constrained_edges = cdt.as_arrays_with_constraints()
    ## "Creates" 3D vertices from 2D cdt vertices (3rd dim is always 0)
    points_3d = np.zeros((vertices.shape[0], 3), dtype=vertices.dtype)
    points_3d[:, :2] = vertices

    # Creates the Surface_mesh
    # FIXME c'est bien vmap, mais est on sur que l'ordre renvoyé par cdt est le
    #       même que l'ordre initial ? Sinon, le problème reste le même...
    mesh, vmap = SM.make_mesh(points_3d, triangles, with_vertices_map=True)

    # Adds the constrained edges to the mesh
    constraints, ok = mesh.add_edge_property("e:is_constrained", dtype="b", value=False)
    assert ok
    for e in constrained_edges:
        constraints[mesh.edge(mesh.halfedge(vmap[e[0]], vmap[e[1]]))] = True

    # Computes the connected components
    nb_comps = PMP.connected_components(mesh, "f:components", constraints)
    return mesh, vmap, nb_comps, mesh.face_property("f:components")


def _compute_components_contours(mesh, nb_comps, prop_comps):
    """
    For each component in the mesh, computes the polyline defining its contour.
    The polyline is an ordered sequence of Vertex_index.
    Returns: a list of list of Surface_mesh::Vertex_index (one list per component)
    """
    comps_contours = []
    # For each component (identified by an integer value)
    for comp_id in range(nb_comps):
        # Get the faces in the component
        faces = [f for f in mesh.faces() if prop_comps[f] == comp_id]
        # Get the halfedges defining the border of the component
        border = PMP.border_halfedges(mesh, faces)
        border = [h for h in border]
        assert border
        # Converts the "soup" of border halfedges to a polyline (an ordered list
        # of vertices) defining the component border
        h = border.pop()
        v_first, v_cur = mesh.source(h), mesh.target(h)
        contour = [v_cur]
        while border:
            # Finds the halfedge whose source matches the target of the previous one
            h = next((he for he in border if mesh.source(he) == v_cur), None)
            assert h is not None
            # Adds its target to the chain
            v_cur = mesh.target(h)
            contour.append(v_cur)
            border.remove(h)
        assert v_cur == v_first
        comps_contours.append(contour)
    return comps_contours


def _reorder_outputs_for_petrel_grid(mesh, vmap, prop_comps, comp_contours):
    # Output 1: 2D vertices ordered as follows:
    #   - the arleady existing vertices (the pillars ones) in their original order
    #   - all the vertices created by the triangulation
    vertices_2d = np.array([[p3d.x, p3d.y] for p3d in (mesh.point(v) for v in vmap)])
    # Output 2: for each mesh face, list of its vertices ids (ie position in vertices_2d)
    vmap = [v for v in vmap]  # We need to easily access element indices
    triangles = np.array(
        [
            [vmap.index(v) for v in SM.vertices_around_face(mesh.halfedge(f), mesh)]
            for f in mesh.faces()
        ]
    )
    # Output 3: for each mesh face, its component id (ordered as triangles)
    # FIXME any difference between using list comprehension and generator expression?
    comp_ids = np.fromiter((prop_comps[f] for f in mesh.faces()), dtype="int")
    # Output 4: for each component (ordered by increasing id), the ordered list of
    # vertex ids defining its closed contour
    contours = [[vmap.index(v) for v in c] for c in comp_contours]
    return vertices_2d, triangles, comp_ids, contours


def mesh(edges_vertices, edges):
    cdt = _build_constrained_delaunay_triangulation(edges_vertices, edges)
    mesh, vmap, nb_comps, prop_comps = _compute_connected_components(cdt)
    comp_contours = _compute_components_contours(mesh, nb_comps, prop_comps)
    return _reorder_outputs_for_petrel_grid(mesh, vmap, prop_comps, comp_contours)
