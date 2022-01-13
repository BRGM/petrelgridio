import numpy as np

from pycgal.Epick import Point_2, Point_3
import pycgal.Mesh_2 as Mesh_2
import pycgal.Polygon_mesh_processing as PMP
import pycgal.Surface_mesh as SM


def _build_constrained_delaunay_triangulation(vertices, edges):
    # Creates empty mesh
    cdt = Mesh_2.Constrained_Delaunay_triangulation_2()
    # Inserts all vertices, registers pairs of (vertex id, vertex handle)
    map_vertices = {i: cdt.insert(Point_2(*v)) for i, v in enumerate(vertices)}
    # Inserts all edges as constraints in the triangulation
    for e in edges:
        cdt.insert_constraint(map_vertices[e[0]], map_vertices[e[1]])
    # Meshes the triangulation
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


def _convert_to_2d_vertices_triangles_and_components(mesh, prop_components):
    # Maps each Vertex_index to an (integer) id
    map_vertices = {i: v_idx for i, v_idx in enumerate(mesh.vertices())}
    # Maps each Halfedge_index to an (integer) id, used to bypass problem with list from Halfedges iterator
    map_halfedges = {i: h_idx for i, h_idx in enumerate(mesh.halfedges())}
    def get_vertex_id(vertex_index):
        """Returns the integer id associated to a Vertex_index"""
        for id, v_index in map_vertices.items():
            if v_index == vertex_index:
                return id
        assert False, "Input Vertex_index does not exists"
    def get_halfedge_id(halfedge_index):
        """Returns the integer id associated to a Halfedge_index"""
        for id, h_index in map_halfedges.items():
            if h_index == halfedge_index:
                return id
        assert False, "Input Vertex_index does not exists"
    def get_point_2d(vertex_idx):
        p3d = mesh.point(vertex_idx)
        return np.array([p3d.x, p3d.y])
    # Maps the 2D points to return to its Vertex_index id
    map_points_2d = {i: get_point_2d(v_idx) for i, v_idx in enumerate(mesh.vertices())}
    vertices_2d = np.zeros((mesh.number_of_vertices(), 2), dtype=np.float64)
    # Creates output 1: array of 2D pillar vertices
    for i in range(mesh.number_of_vertices()):
        vertices_2d[i,:] = map_points_2d[i]
    # Creates output 2: for each 2D pillars triangle: list of vertices ids
    triangles_2d = np.array(
        [[get_vertex_id(v) for v in SM.vertices_around_face(f, mesh)] for f in mesh.faces()]
    )
    # Creates output 3: for each 2D pillars triangle: id of the component it belongs to
    components_2d = np.array([prop_components[f] for f in mesh.faces()])
    # Determines for each component the list of all its halfedges that are on the component border
    components_borders = []
    halfedges_on_mesh_border = PMP.border_halfedges(mesh) # FIXME Needed while mesh.is_border(halfedge) is not available
    for comp_id in range(np.max(components_2d) + 1): # For each component (comp_id starts at 0)
        # Get list of faces in the component
        component_faces = [
            f for f in mesh.faces() if prop_components[f] == comp_id
        ]
        # Get list of all halfedges on component border
        border = []
        # For each edge, find halfedges on component border
        for f in component_faces: # For each face in the component
            for h in SM.halfedges_around_face(mesh.halfedge(f), mesh): # For each face halfedge
                assert h not in halfedges_on_mesh_border, "What ?! This face halfedge cannot point to a null_face, can it ?"
                h_opposite = mesh.opposite(h)
                if h_opposite in halfedges_on_mesh_border: # FIXME Replace by mesh.is_border(h_opposite) once available
                    border.append(get_halfedge_id(h)) # Face is on mesh border
                elif prop_components[f] != prop_components[mesh.face(h_opposite)]:
                    border.append(get_halfedge_id(h)) # Adjacent face is in another component
        assert len(border) > 2, "A closed 2D border should have at least 3 halfedges"
        assert len(border) == len(set(border)), "Border contains duplicated elements"
        components_borders.append(border)
    # Creates output 4: for each component: list of vertices ids defining the component border
    components_borders_2d = [] 
    # Converts list of halfedges to a "chained" list of vertices
    for border in components_borders: # For each component border
        # Work with Halfedges_index rather than their id
        border = [map_halfedges[h_id] for h_id in border] 
        # Treats the first halfedge
        h = border.pop()
        first_v_index, cur_v_index = mesh.source(h), mesh.target(h)
        chain = [get_vertex_id(first_v_index), get_vertex_id(cur_v_index)]
        # Treats others
        while border: # While border is not empty
            # Finds halfedge whose source vertex is the current last vertex of the chain
            h = next((he for he in border if mesh.source(he) == cur_v_index), None)
            assert h is not None
            cur_v_index = mesh.target(h)
            # Adds its target vertex to the chain
            chain.append(get_vertex_id(cur_v_index))
            border.remove(h)
        assert chain[0] == chain[-1], "The border is not a closed curve"
        assert cur_v_index == first_v_index, "The border is not a closed curve" # Just for debug 
        components_borders_2d.append(chain[:-1]) # Remove the duplicated vertex
    return vertices_2d, triangles_2d, components_2d, components_borders_2d


def _report_defect(mesh, prop_components):
    """
    Problem: Cannot create a list from iteror SM.halfedges_around_face(mesh.halfedge(f), mesh).
    I guess the list stores "the adress of the iterator, and not its value. Result:
        l = []
        for h in SM.halfedges_around_face(mesh.halfedge(f), mesh):
            l.append(h)
    Result after 1 iteration(s): [Halfedge_index(0)]
    Result after 2 iteration(s): [Halfedge_index(1), Halfedge_index(1)]
    The value of the 1st list element changed!
    """
    # Creates output 3: for each 2D pillars triangle: id of the component it belongs to
    components_2d = np.array([prop_components[f] for f in mesh.faces()])
    # Creates output 4: for each component: list of bounding vertices
    components_borders = []
    border_halfedges = PMP.border_halfedges(mesh) # FIXME Needed while mesh.is_border(halfedge) is not available
    for comp_id in range(np.max(components_2d)): # For each component
        print("Component ", comp_id)
        # Get list of faces in the component
        component_faces = [
            f for f in mesh.faces() if prop_components[f] == comp_id
        ]
        print("  Faces in component:", component_faces)
        # Get list of all halfedges on component border
        comp_border_halfedges = []
        print("  Created new comp_border_halfedges:", comp_border_halfedges)
        # For each edge, find halfedges on component border
        for f in component_faces: # For each face in the component
            print("    Face:", f)
            for h in SM.halfedges_around_face(mesh.halfedge(f), mesh):
                print("      Halfedge:", h)
                print("        Current state:", comp_border_halfedges)
                assert h not in border_halfedges, "What ?! This face halfedge cannot point to a null_face, can it ?"
                h_opposite = mesh.opposite(h)
                if h_opposite in border_halfedges: # FIXME Replace by mesh.is_border(h_opposite) once available
                    # print("        Current state:", comp_border_halfedges)
                    comp_border_halfedges.append(h) # Face is on mesh border
                    # print("        Updated state (mesh):", comp_border_halfedges)
                elif prop_components[f] != prop_components[mesh.face(h_opposite)]:
                    # print("        Current state:", comp_border_halfedges)
                    comp_border_halfedges.append(h) # Adjacent face is in another component
                    # print("        Updated state (comp):", comp_border_halfedges)
                # else:
                #     print("        Nothing to do")
                print("        End     state:", comp_border_halfedges)
                
        components_borders.append(comp_border_halfedges)


def mesh(edges_vertices, edges):
    # 2D triangulation
    cdt = _build_constrained_delaunay_triangulation(edges_vertices, edges)
    mesh = _convert_triangulation_to_surface_mesh(cdt)
    prop_components = _compute_connected_components(mesh, edges, edges_vertices)
    #############################################################
    # FIXME Uncomment to reproduce the problem
    # return _report_defect(mesh, prop_components)
    #############################################################
    return _convert_to_2d_vertices_triangles_and_components(mesh, prop_components)
