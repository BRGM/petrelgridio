# To do list

## Create a proper packaging

## Enforce np.dtype wherever possible
 * In mesh classes: HexMesh, TetMesh, (HybridMesh?, RawMesh?)
 * * `self._vertices = np.array((NbVertices, Elt.DIMENSION=0), dtype = np.float)`
 * * `self._cells = np.array((NbVertices, Elt.NBFACES=0), dtype = np.uint64)`
 * In PetrelGrid?