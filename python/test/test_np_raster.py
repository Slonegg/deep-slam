from deep_slam.utils.np_raster import block_triangles, rasterize
from deep_slam.utils.projection import ortho
import numpy as np


def test_block_triangles():
    vertices = [[0.5, 2.5], [1.5, 5.5], [2.5, 3.5],
                [3.5, 2.5], [5.5, 2.5], [4.5, 1.5],
                [1.5, 4.5], [4.5, 5.5], [3.5, 4.5]]
    indices = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    w = 7
    h = 6

    # convert vertices to clip space
    vertices = np.array(vertices)
    vertices /= np.array([w, h])
    vertices *= 2.0
    vertices -= 1.0

    bi, bt = block_triangles(vertices, indices, width=w, height=h, block_size=(1, 1))

    # TODO: fix suboptimal block_triangles
    # cells that triangles intersect
    cells = [[14, 15, 16, 21, 22, 23, 28, 29, 30, 25, 35, 36, 37],
             [10, 11, 12, 17, 18, 19],
             [29, 30, 31, 32, 36, 37, 38, 39]]
    empty_cells = [0, 1, 2, 3, 4, 5, 6, 7, 25, 26, 27, 40, 41]
    for i, t in zip(bi, bt):
        assert i in cells[t]
        assert i not in empty_cells


def test_rasterize_zbuffer():
    w = 240
    h = 240
    red = [1, 0, 0]
    green = [0, 1, 0]
    vertices = [[w/3, h/3, 0], [w/3, 2*h/3, 0], [2*w/3, 2*h/3, 0], [2*w/3, h/3, 0],
                [0, 0, 1], [0, h, 1], [w, h, 1], [w, 0, 1]]
    colors = [red, red, red, red,
              green, green, green, green]
    indices = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]
    proj = ortho(0, w, 0, h, 0, 10)

    img = rasterize({'position': vertices, 'color': colors}, indices, mvp=proj, width=w, height=h)

    assert np.all(img[w//2, h//2] == red)
    assert np.all(img[3*w//4, 3*h//4] == green)
    assert np.all(img[w//4, h//4] == green)


def test_rasterize_triangles():
    vertices = [[60, 100, 1], [120, 200, 1], [160, 120, 1],
                [160, 100, 1], [300, 120, 1], [240, 40, 1]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
              [1, 1, 0], [0, 1, 1], [1, 0, 1]]
    indices = [[0, 1, 2], [3, 4, 5]]
    w = 320
    h = 240
    proj = ortho(0, w, 0, h, 0, 1)

    img = rasterize({'position': vertices, 'color': colors}, indices, mvp=proj, width=w, height=h)

    assert np.any(img[110, 80] != 0)
    assert np.any(img[150, 180] != 0)
    assert np.all(img[10, 10] == 0)
