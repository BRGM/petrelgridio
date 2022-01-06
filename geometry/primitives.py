import numpy as np

class Point3d:
    def __init__(self, xyz):
        self.coords = xyz
    
    @classmethod
    def origin(cls):
        return Point3d(0., 0., 0.)
    
    @property
    def coords(self):
        return self._coords
        
    @coords.setter
    def coords(self, other):
        assert(other.ndim == 1)
        assert(other.size == 3)
        assert(other.dtype == np.float64)
        self._coords = np.copy(other) # FIXME Necessary? Keep for security?
    
    def __iadd__(self, other):
        assert isinstance(other, Point3d)
        self._coords += other.coords


class Primitive:
    DIMENSION = 3 # We work exclusively in a 3D space
    NB_VERTICES = None
    VTK_ELEMENT_ID = None

    @classmethod
    def is_valid(cls):
        """Used to forbid instantiation of incomplete parent types Primitive(*d)"""
        return cls.NB_VERTICES is not None and \
               cls.VTK_ELEMENT_ID is not None
    
    def __init__(self, vertices_ids):
        assert self.is_valid(), "Instanciation of undefined primitive type is forbidden"
        assert vertices_ids.ndim == 1, "Input array should be 1D"
        assert vertices_ids.size == self.NB_VERTICES,  "Input array should have 'cls.NB_VERTICES' elements"
        assert vertices_ids.dtype == np.int64, "Input array should store integer (numpy.int64) indices"
        self._vertices = np.copy(vertices_ids)
        
    def __hash__(self):
        return hash((self.__class__.__name__, *self.vertices))
    
    def __eq__(self, other):
        return hash(self) == hash(other)
    
    @property
    def vertices(self):
        return self._vertices


class Primitive2d(Primitive):
    def __init__(self, vertices_ids):
        super().__init__(vertices_ids)

    @staticmethod
    def get_face_from_facet(facet):
        """
        Returns a new Primitive2d having the same type and vertices as the input
        facet, but with vertices ordered differently.
        New vertices order: the new 1st vertex is the one with the minimum index,
        the 2nd one is its neighbor with the lowest index, and we continue cycling
        until all vertices have been treated.
        Example: 
        Given facet_vertices = [V0, V1, ..., Vi , ..., Vn] with Vi = min(facet_vertices)
        If Vi-1 >= Vi+1:
            face_vertices = [Vi, Vi+1, ..., Vn, V0, V1, ..., Vi-1]
        Else:
            face_vertices = [Vi, Vi-1, ..., V1, V0, Vn, ..., Vi+1]
        """
        assert isinstance(facet, Primitive2d)
        vertices = facet.vertices
        idx = vertices.argmin()
        tmp = None
        if vertices[idx - 1] < vertices[(idx + 1) % vertices.size]:
            tmp = np.flip(np.roll(vertices, vertices.size - (1 + idx)))
        else:
            tmp = np.roll(vertices, -idx)
        assert tmp[0] == np.min(tmp)
        assert tmp[1] < tmp[-1]
        return type(facet)(tmp)


class Triangle(Primitive2d):
    NB_VERTICES = 3
    VTK_ELEMENT_ID = 5
    
    def __init__(self, vertices_ids):
        super().__init__(vertices_ids)


class Quad(Primitive2d):
    NB_VERTICES = 4
    VTK_ELEMENT_ID = 9
    
    def __init__(self, vertices_ids):
        super().__init__(vertices_ids)


class Primitive3d(Primitive):
    NB_FACES = None
    
    def __init__(self, vertices_ids):
        super().__init__(vertices_ids)

    def facets(self): # FIXME Move to class Primitive: 2D primitives have Edge "facets"
        assert False, "Should be reimplemented in each derived type"

    @staticmethod
    def builder(vertices_ids):
        """Returns the 3D primitive matching the size of vertices"""
        assert vertices_ids.ndim == 1
        assert vertices_ids.dtype == np.int64

        size = vertices_ids.size
        if size == 8:
            return Hexahedron(vertices_ids)
        elif size == 4:
            return Tetrahedron(vertices_ids)
        elif size == 5:
            return Pyramid(vertices_ids)
        elif size == 6:
            return Wedge(vertices_ids)
        else:
            assert False, "Number of vertices does not match any available 3D primitive type"


class Tetrahedron(Primitive3d):
    NB_VERTICES = 4
    NB_FACES = 4
    VTK_ELEMENT_ID = 10

    def __init__(self, vertices_ids):
        super().__init__(vertices_ids)

    def facets(self):
        return [
                Triangle(np.array([self.vertices[1], self.vertices[2], self.vertices[3]])),
                Triangle(np.array([self.vertices[0], self.vertices[3], self.vertices[2]])),
                Triangle(np.array([self.vertices[0], self.vertices[1], self.vertices[3]])),
                Triangle(np.array([self.vertices[0], self.vertices[2], self.vertices[1]]))
        ]


class Hexahedron(Primitive3d):
    NB_VERTICES = 8
    NB_FACES = 6
    VTK_ELEMENT_ID = 12

    def __init__(self, vertices_ids):
        super().__init__(vertices_ids)

    def facets(self):
        return [
                Quad(np.array([self.vertices[0], self.vertices[1], self.vertices[2], self.vertices[3]])),
                Quad(np.array([self.vertices[4], self.vertices[5], self.vertices[6], self.vertices[7]])),
                Quad(np.array([self.vertices[1], self.vertices[2], self.vertices[6], self.vertices[5]])),
                Quad(np.array([self.vertices[2], self.vertices[6], self.vertices[7], self.vertices[3]])),
                Quad(np.array([self.vertices[3], self.vertices[7], self.vertices[4], self.vertices[0]])),
                Quad(np.array([self.vertices[0], self.vertices[1], self.vertices[5], self.vertices[4]]))
        ]


class Pyramid(Primitive3d):
    NB_VERTICES = 5
    NB_FACES = 5
    VTK_ELEMENT_ID = 14

    def __init__(self, vertices_ids):
        super().__init__(vertices_ids)

    def facets(self):
        return [
                Quad(np.array([self.vertices[0], self.vertices[1], self.vertices[2], self.vertices[3]])),
                Triangle(np.array([self.vertices[0], self.vertices[1], self.vertices[4]])),
                Triangle(np.array([self.vertices[1], self.vertices[2], self.vertices[4]])),
                Triangle(np.array([self.vertices[2], self.vertices[3], self.vertices[4]])),
                Triangle(np.array([self.vertices[3], self.vertices[0], self.vertices[4]]))
        ]


class Wedge(Primitive3d):
    NB_VERTICES = 6
    NB_FACES = 5 # FIXME Useful?
    VTK_ELEMENT_ID = 13

    def __init__(self, vertices_ids):
        super().__init__(vertices_ids)

    def facets(self):
        return [
                Triangle(np.array([self.vertices[0], self.vertices[1], self.vertices[2]])),
                Triangle(np.array([self.vertices[3], self.vertices[4], self.vertices[5]])),
                Quad(np.array([self.vertices[0], self.vertices[2], self.vertices[5], self.vertices[3]])),
                Quad(np.array([self.vertices[1], self.vertices[4], self.vertices[5], self.vertices[2]])),
                Quad(np.array([self.vertices[0], self.vertices[3], self.vertices[4], self.vertices[1]]))
        ]
