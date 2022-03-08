import numpy as np


def _get_v1():
    return np.array(
        [
            [0.0, 1.00],
            [0.0, 0.90],
            [0.0, 0.80],
            [0.0, 0.70],
            [0.0, 0.60],
            [0.0, 0.30],
            [0.0, 0.10],
            [0.0, 0.00],
            [1.0, 1.00],
            [1.0, 0.90],
            [1.0, 0.70],
            [1.0, 0.50],
            [1.0, 0.30],
            [1.0, 0.20],
            [1.0, 0.15],
            [1.0, 0.00],
        ],
        dtype=np.float64,
    )


def _get_v2():
    return np.array(
        [[0.0, 0.75], [0.0, 0.55], [0.0, 0.35], [1.0, 0.85], [1.0, 0.75], [1.0, 0.65]],
        dtype=np.float64,
    )


def _get_e1():
    return np.array([(i, i + 8) for i in range(8)], dtype=np.int32)


def _get_e2():
    return np.array([(i, i + 3) for i in range(3)], dtype=np.int32)


def get_fault_pillars_data():
    v1 = _get_v1()
    v2 = _get_v2()
    e1 = _get_e1()
    e2 = _get_e2() + len(v1)
    return np.concatenate([v1, v2]), np.concatenate([e1, e2])


if __name__ == "__main__":
    # Simple main to visualize test data using matplotlib
    import matplotlib.pyplot as plt

    v1, v2 = _get_v1(), _get_v2()
    e1, e2 = _get_e1(), _get_e2()
    plt.xlim(-1, 2), plt.ylim(-0.1, 1.1)  # Set X,Y plot boundaries
    plt.vlines((0, 1), 0, 1, colors="black")
    plt.scatter([v[0] for v in v1], [v[1] for v in v1], color="blue")
    plt.scatter([v[0] for v in v2], [v[1] for v in v2], color="orange")
    for e in e1:
        p0, p1 = v1[e[0]], v1[e[1]]
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], color="blue")
    for e in e2:
        p0, p1 = v2[e[0]], v2[e[1]]
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], color="orange")
    plt.show()
