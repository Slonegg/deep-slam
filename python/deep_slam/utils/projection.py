import numpy as np


def ortho(l, r, b, t, n, f, dtype=np.float32):
    return np.array([[2 / (r - l), 0, 0, (r+l)/(l-r)],
                     [0, 2 / (t - b), 0, (t+b)/(b-t)],
                     [0, 0, 2 / (n - f), (f+n)/(n-f)],
                     [0, 0, 0, 1]], dtype=dtype)


def perspective(fovy, aspect, znear, zfar, dtype=np.float32):
    f = np.tan(2 / fovy)
    return np.array([[f / aspect, 0, 0, 0],
                     [0, f, 0, 0],
                     [0, 0, (znear + zfar) / (znear - zfar), 2 * znear * zfar / (znear - zfar)],
                     [0, 0, -1, 0]], dtype=dtype)
