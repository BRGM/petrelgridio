# PetrelGridIO
Exports a Petrel pillar grid (`.grdecl` file) into a `.vtu` file that can be loaded in Paraview.

## Installation
### Requirements
`petrelgridio` depends on:
* vtkwriters >= 0.0.6
* pyCGAL >= 0.2.18
* verstr >= 0.1.2

### Install from PyPI
```bash
python -m pip install verstr vtkwriters pycgal petrelgridio
```

## Basic example
```python
from petrelgridio.petrel_grid import PetrelGrid
from petrelgridio.raw_mesh import RawMesh
from petrelgridio.vtu import to_vtu

filename_in = "path/to/file.grdecl"
filename_out = "where/to/store/file.vtu"
grid = PetrelGrid.build_from_files(filename_in)
hexa, _ = grid.process()
vertices, cells_faces, faces_nodes = grid.process_faults(hexa)
mesh = RawMesh(vertices, faces_nodes, cells_faces)
to_vtu(mesh, filename_out)
```

## Authors
* Jean-Pierre Vergnes (BRGM - DEPA/GDR)
* Nicolas Clausolles (BRGM - DNG/TIA)
* Simon Lopez (BRGM - DGR/CIM)
