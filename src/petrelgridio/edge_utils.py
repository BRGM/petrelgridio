from collections import defaultdict
import copy

# to be used as unique ids edge vertices are sorted
Edge = lambda i, j: (j, i) if i > j else (i, j)


class Edge_faces:
    def __init__(self, faces=None):
        self.edge_faces = defaultdict(list)
        if faces is not None:
            self.register_faces(faces)

    def register_faces(self, faces):
        edge_faces = self.edge_faces
        for fi, nodes in enumerate(faces):
            for i in range(len(nodes)):
                edge_faces[Edge(nodes[i - 1], nodes[i])].append(fi)
        # edge_faces.default_factory = None # reset dict behavior (will raise KeyError) - does not work ?

    def seems_consistent(self):
        return all([len(faces) for faces in self.edge_faces.values()])

    @property
    def nb_edges(self):
        return len(self.edge_faces)

    def edges(self):
        return self.edge_faces.keys()


def edge_is_in_face(edge, face_nodes):
    return any(
        edge == Edge(face_nodes[i - 1], face_nodes[i]) for i in range(len(face_nodes))
    )


def locate_edge_in_face(edge, face_nodes):
    for i in range(len(face_nodes)):
        if edge == Edge(face_nodes[i - 1], face_nodes[i]):
            return i
    raise ValueError("edge {} not in face {}".format(str(edge), str(face_nodes)))


def replace_edge(face, old_edge, new_edges):
    i = locate_edge_in_face(old_edge, face)
    new_face = [face[k] for k in range(i + 1, len(face) + min(0, i - 1))]
    for k in range(max(0, i - 1)):
        new_face.append(face[k])
    nk = face[i - 1]
    if nk not in new_edges[0]:
        new_edges = new_edges[::-1]
    assert nk in new_edges[0]
    new_face.append(nk)
    for new_edge in new_edges:
        n1, n2 = new_edge
        if nk == n1:
            new_face.append(n2)
        else:
            assert nk == n2
            new_face.append(n1)
        nk = new_face[-1]
    assert nk == face[i]
    return new_face
