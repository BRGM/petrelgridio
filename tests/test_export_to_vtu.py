from geometry.petrel_grid import PetrelGrid
from geometry.hex_mesh import HexMesh
from geometry.hybrid_mesh import HybridMesh
from geometry.raw_mesh import RawMesh
from ios.vtu import to_vtu

import geometry.dummy_petrel_grids as dg


data_folder = "/mnt/d/Documents/clausolles/Workspace/TESTS/PetrelGridIO/tests/data/"


def test_export_to_vtu():
    ##### Test models #####

    # name = "20X20X6.GRDECL"
    # name = "Simple3x3x1.grdecl"
    # name = "Simple3x3x3.grdecl"
    # name = "Simple20x20x5.grdecl"

    ##### Build grid #####
    name = "Simple20x20x5_Fault.grdecl" # FIXME Pourquoi ça marche ?
    grid = PetrelGrid.build_from_files(data_folder + name)

    ##### Build hex mesh #####
    hexa, vertices, cell_faces, face_nodes = grid.process()
    mesh = HexMesh.make(vertices, hexa) 
    to_vtu(mesh, name)

    ##### Build raw mesh #####
    raw_mesh = RawMesh(vertices=vertices, face_nodes=face_nodes, cell_faces=cell_faces)
    hybrid_mesh, original_cell = raw_mesh.as_hybrid_mesh()
    print(
        f"Splitted {name} mesh with: {hybrid_mesh.nb_vertices} vertices, {hybrid_mesh.nb_cells} cells, {hybrid_mesh.nb_faces} faces"
    )
    # FIXME Start here
    to_vtu(hybrid_mesh, f"{name}_splitted", celldata={"original_cell": original_cell})

    ##### Build with faults #####
    # TODO 


def test_dummy_grid():
    # Test 1: common_node
    name = "common_node"
    print(f"  test 1: {name}")
    hexahedra = dg.common_node()
    pillars = dg.pillars(hexahedra)
    grid = PetrelGrid.build_from_arrays__for_dummy_grids(pillars[..., :3], pillars[..., 3:], hexahedra[..., 2])
    
    hexa, vertices, cell_faces, face_nodes = grid.process()
    dg.depth_to_elevation(vertices) # Je ne sais pas pourquoi, mais c'est dans le test de MeshTools...
    mesh = HexMesh.make(vertices, hexa)
    to_vtu(mesh, name)
    
    # Test 2: sugar_box
    name = "sugar_box"
    print(f"  test 2: {name}")
    hexahedra = dg.grid_of_heaxaedra((4, 3, 2))
    pillars = dg.pillars(hexahedra)
    grid = PetrelGrid.build_from_arrays__for_dummy_grids(pillars[..., :3], pillars[..., 3:], hexahedra[..., 2])
    
    hexa, vertices, cell_faces, face_nodes = grid.process()
    dg.depth_to_elevation(vertices) # Je ne sais pas pourquoi, mais c'est dans le test de MeshTools...
    mesh = HexMesh.make(vertices, hexa)
    to_vtu(mesh, name)
    
    # Test 3: stairs
    name = "stairs"
    print(f"  test 3: {name}")
    hexahedra = dg.four_cells_stairs()
    pillars = dg.pillars(hexahedra)
    grid = PetrelGrid.build_from_arrays__for_dummy_grids(pillars[..., :3], pillars[..., 3:], hexahedra[..., 2])
    
    hexa, vertices, cell_faces, face_nodes = grid.process()
    dg.depth_to_elevation(vertices) # Je ne sais pas pourquoi, mais c'est dans le test de MeshTools...
    mesh = HexMesh.make(vertices, hexa)
    to_vtu(mesh, name)
    
    # Test 4: ramp
    name = "ramp"
    print(f"  test 4: {name}")
    hexahedra = dg.faulted_ramp((8, 2, 1), begin=0.33)
    pillars = dg.pillars(hexahedra)
    grid = PetrelGrid.build_from_arrays__for_dummy_grids(pillars[..., :3], pillars[..., 3:], hexahedra[..., 2])
    
    hexa, vertices, cell_faces, face_nodes = grid.process()
    dg.depth_to_elevation(vertices) # Je ne sais pas pourquoi, mais c'est dans le test de MeshTools...
    mesh = HexMesh.make(vertices, hexa)
    to_vtu(mesh, name)