# PetrelGridIO
Exports a Petrel pillar grid (.grdecl file) into a .vtu file that can be loaded in Paraview.

## Requirements
PetrelGridIO (directly) depends on:
* [vtkwriters](https://gitlab.brgm.fr/brgm/modelisation-geologique/vtkwriters) >= 0.0.6
* [pyCGAL](https://gitlab.brgm.fr/brgm/geomodelling/internal/pycgal) >= 0.2.18

Note that these libraries have their own dependencies that you need to install too (see their `README.md` for installing them).

## Getting started
Once you have installed `vtkwriters`, `pycgal` and their dependencies, you can install `petrelgridio`:
```bash
# Install the pre-built wheel provided on the BRGM Nexus
python -m pip install petrelgridio -i https://nexus.brgm.fr/repository/pypi-all/simple
```

Alternatively, if you wish to install `petrelgridio` from sources, you can use the following:
```bash
# Install from sources
git clone https://gitlab.brgm.fr/brgm/modelisation-geologique/essais/nicolas/petrelgridio
cd petrelgridio
python -m pip install .
```

## Authors
* Nicolas Clausolles (DNG/TIA)
* Simon Lopez (DGR/CIM)

## Bug reports & improvements
Use GitLab [issues](https://gitlab.brgm.fr/brgm/modelisation-geologique/essais/nicolas/petrelgridio/-/issues)
