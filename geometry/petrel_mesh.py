import numpy as np

import pycgal.Mesh_2 as Mesh_2
from pycgal.Epick import Point_2 as Point


def _build_constrained_delaunay_triangulation(vertices, edges):
    cdt = Mesh_2.Constrained_Delaunay_triangulation_2()
    # FIXME Assumes no duplicated v index in segments
    for e in edges:
        v0 = cdt.insert(Point(*vertices[e[0]]))
        v1 = cdt.insert(Point(*vertices[e[1]]))
        cdt.insert_constraint(v0, v1)
    # Meshing the triangulation
    # FIXME No adaptive critrion yet?
    criteria = Mesh_2.Delaunay_mesh_adaptative_size_criteria_2(S=0.5)
    Mesh_2.refine_Delaunay_mesh_2(cdt, criteria)
    Mesh_2.Delaunay_mesher_2(cdt)
    return cdt


def mesh(vertices, edges):
    cdt = _build_constrained_delaunay_triangulation(vertices, edges)
    print(cdt.as_arrays())