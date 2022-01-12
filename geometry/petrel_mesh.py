import numpy as np

from pycgal.Epick import Point_2, Point_3
import pycgal.Mesh_2 as Mesh_2
import pycgal.Polygon_mesh_processing as PMP
import pycgal.Surface_mesh as SM

def _build_constrained_delaunay_triangulation(vertices, edges):
    cdt = Mesh_2.Constrained_Delaunay_triangulation_2()
    # FIXME Assumes no duplicated v index in segments
    for e in edges:
        v0 = cdt.insert(Point_2(*vertices[e[0]]))
        v1 = cdt.insert(Point_2(*vertices[e[1]]))
        cdt.insert_constraint(v0, v1)
    # Meshing the triangulation
    # FIXME No adaptive critrion yet?
    criteria = Mesh_2.Delaunay_mesh_adaptative_size_criteria_2(S=0.5)
    Mesh_2.refine_Delaunay_mesh_2(cdt, criteria)
    Mesh_2.Delaunay_mesher_2(cdt)
    assert cdt.number_of_faces() > 0
    return cdt


def _convert_triangulation_to_surface_mesh(cdt):
    vertices, triangles = cdt.as_arrays()
    # "Creates" 3D vertices from 2D cdt vertices (3rd dim is always 0)
    vertices_3d = np.zeros((vertices.shape[0], 3), dtype=vertices.dtype)
    vertices_3d[:,:2] = vertices
    return SM.Surface_mesh(vertices_3d, triangles)
    

def _compute_connected_components(mesh, constrained_edges, vertices_2d):
    # Adds constrained edges property on mesh
    constraints, created = mesh.add_edge_property(
        "e:is_constrained", dtype="b", value=False
    )
    assert created
    # Finds 3D mesh vertices corresponding to 2D pillars nodes
    points_3d = [Point_3(v[0], v[1], 0.) for v in vertices_2d]
    vertices_3d = SM.locate(mesh, points_3d) # Ordered as vertices_2d
    assert len(points_3d) == len(vertices_3d)
    # Labels all 3D mesh edges corresponding to 2D pillar segments
    constrained_mesh_edges = [[vertices_3d[e[0]], vertices_3d[e[1]]] for e in constrained_edges]
    for he in mesh.halfedges():
        v0, v1 = mesh.source(he), mesh.target(he)
        if [v0, v1] in constrained_mesh_edges:
            constraints[mesh.edge(he)] = True
    # Computes (and labels) components
    PMP.connected_components(mesh, "f:components", constraints)
    return mesh.face_property("f:components")
    # print(prop_components.copy_as_array())
    

def mesh(edges_vertices, edges):
    # 2D triangulation
    cdt = _build_constrained_delaunay_triangulation(edges_vertices, edges)
    mesh = _convert_triangulation_to_surface_mesh(cdt)
    prop_components = _compute_connected_components(mesh, edges, edges_vertices)
    # TODO Continue...


def OLD__mesh(edges_vertices, edges):
    cdt = _build_constrained_delaunay_triangulation(edges_vertices, edges)
    assert cdt.number_of_faces() > 0
    # Total number of vertices (some might have appeared during triangulation)
    nb_vertices = cdt.number_of_vertices()
    # All 2D vertices (i.e., projected onto the fault plane)
    vertices = np.ndarray((nb_vertices, 2), dtype=np.float64)
    vmap = {}
    # Filling of vertices
    nb_added_vertices = edges_vertices.shape[0] # Number of vertices added to vertices
    assert nb_vertices >= nb_added_vertices
    vertices[:nb_added_vertices] = edges_vertices # Copy original vertices
    # Add newly created vertices

    cdt_vertices, cdt_trgls = cdt.as_arrays()
    print(f"cdt_vertices, size = {cdt_vertices.shape[0]}")
    print(cdt_vertices)
    print(f"cdt_trgls, size = {cdt_trgls.shape[0]}")
    print(cdt_trgls)
