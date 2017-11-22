import numpy as np


def _unwrap_xyz(args):
    if len(args) == 1:
        return args[0][0], args[0][1], args[0][2]
    elif len(args) == 3:
        return args[0], args[1], args[2]
    else:
        raise ValueError('Unsupported number of arguments')


def translation(*args, dtype=np.float32):
    x, y, z = _unwrap_xyz(args)
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]], dtype=dtype)


def rotation(angle, *args, dtype=np.float32):
    x, y, z = _unwrap_xyz(args)
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[x*x*(1-c)+c, x*y*(1-c)-z*s, x*z*(1-c)+y*s, 0],
                     [y*x*(1-c)+z*s, y*y*(1-c)+c, y*z*(1-c)-x*s, 0],
                     [x*z*(1-c)-y*s, y*z*(1-c)+x*s, z*z*(1-c)+c, 0],
                     [0, 0, 0, 1]], dtype=dtype)
