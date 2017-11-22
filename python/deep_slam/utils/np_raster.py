import numpy as np


def barycentric(p, t):
    """
    For each point p find its barycentric coordinates s, t in triangle t.
    :param p: array of points of shape (height, width, 2)
    :param t: array of triangle vertices of shape (height, width, 3, 2..4)
    :return: boolean array of shape (height, width), entry is true if point p[i, j] is in triangle t[i, j]
    """
    A = -t[:, :, 1, 1] * t[:, :, 2, 0] + t[:, :, 0, 1] * (-t[:, :, 1, 0] + t[:, :, 2, 0])\
        + t[:, :, 0, 0] * (t[:, :, 1, 1] - t[:, :, 2, 1]) + t[:, :, 1, 0] * t[:, :, 2, 1]
    u = t[:, :, 0, 1] * t[:, :, 2, 0] - t[:, :, 0, 0] * t[:, :, 2, 1]\
        + (t[:, :, 2, 1] - t[:, :, 0, 1]) * p[:, :, 0] + (t[:, :, 0, 0] - t[:, :, 2, 0]) * p[:, :, 1]
    u *= np.sign(A)
    v = t[:, :, 0, 0] * t[:, :, 1, 1] - t[:, :, 0, 1] * t[:, :, 1, 0]\
        + (t[:, :, 0, 1] - t[:, :, 1, 1]) * p[:, :, 0] + (t[:, :, 1, 0] - t[:, :, 0, 0]) * p[:, :, 1]
    v *= np.sign(A)

    mask = np.logical_and(u >= 0, v >= 0)
    mask = np.logical_and(mask, (u + v) < np.abs(A))

    A = np.maximum(np.abs(A), np.finfo(np.float32).eps)
    u /= A
    v /= A

    return u, v, mask


def persp_interp(b, vertex_inv_z, pixel_inv_z, vertex_attr):
    """
    Perform perspective corrected interpolation in screen space.
    See [here](interpolation https://www.comp.nus.edu.sg/~lowkl/publications/lowk_persp_interp_techrep.pdf)
    :param u: array of barycentric coordinates corresponding first triangle vertex of shape (height, width)
    :param v: array of barycentric coordinates corresponding second triangle vertex of shape (height, width)
    :param z: array of triangle vertex z coordinates of shape (height, width, 3)
    :param x: array of interpolant values at triangle vertices of shape (height, width, n)
    :return: array of interpolant values at pixels of shape (height, width, n)
    """
    if vertex_attr.ndim == 4:
        return np.moveaxis(np.sum(np.moveaxis(vertex_attr, 3, 0) * b * vertex_inv_z, axis=-1) / pixel_inv_z, 0, 2)
    elif vertex_attr.ndim == 3:
        return np.sum(vertex_attr * b * vertex_inv_z, axis=-1) / pixel_inv_z
    else:
        raise RuntimeError('Unexpected number of dimensions of attribute array')


def block_rasterize(color_buffer, depth_buffer, vertices, attributes, indices, triangle_indices, width, height, block_size):
    # coordinates of the triangles in the block, (height, width, 3, xy)
    nx, ny = width // block_size[0], height // block_size[1]
    inds_map = np.zeros((height, width), dtype=np.int32)
    for i in range(ny):
        for j in range(nx):
            oy = block_size[1]*i
            ox = block_size[0]*j
            inds_map[oy:oy+block_size[1], ox:ox+block_size[0]] = i*nx + j
    triangles = vertices[indices[triangle_indices[inds_map]]]

    # pixel coordinates, (height, width, xy)
    x = np.arange(0, width) + 0.5
    y = np.arange(0, height) + 0.5
    points = np.swapaxes(np.array(np.meshgrid(x, y)).T, 0, 1)

    # find barycentric coordinates
    s, t, mask = barycentric(points, triangles)
    bar = np.dstack((1 - s - t, s, t))

    # perform z-test
    vertex_inv_z = 1 / np.maximum(triangles[:, :, :, 2], np.finfo(np.float32).eps)
    pixel_inv_z = np.sum(bar * vertex_inv_z, axis=-1)
    mask = np.logical_and(mask, pixel_inv_z <= depth_buffer)

    # run pixel shader
    varyings = {}
    for name, attribute in attributes.items():
        if name not in ('position',):
            x = attribute[indices[triangle_indices[inds_map]]]
            varyings[name] = persp_interp(bar, vertex_inv_z, pixel_inv_z, x)

    color_buffer[mask] = varyings['color'][mask] if 'color' in varyings else 255
    depth_buffer[mask] = pixel_inv_z[mask]


def split_viewport(viewport, axis, split):
    v0 = viewport.copy()
    v0[1, axis] = split
    v1 = viewport.copy()
    v1[0, axis] = split
    return v0, v1


def split_triangles(vertices, indices, triangles, viewport, block_size):
    viewport = np.array(viewport)
    viewport_size = np.array((viewport[1, 0] - viewport[0, 0], viewport[1, 1] - viewport[0, 1]))
    if all(viewport_size == block_size):
        return [(viewport, triangles)]

    # split viewport
    axis = 0 if viewport_size[0] > viewport_size[1] else 1
    split = (viewport[0, axis] + viewport[1, axis]) / 2
    v0, v1 = split_viewport(viewport, axis, split)

    # split triangles
    verts = vertices[indices[triangles]]
    tri_l = triangles[np.any([verts[:, i, axis] < split for i in range(3)], axis=0)]
    tri_r = triangles[np.any([verts[:, i, axis] >= split for i in range(3)], axis=0)]

    return split_triangles(vertices, indices, tri_l, v0, block_size)\
        + split_triangles(vertices, indices, tri_r, v1, block_size)


def triangles_in_viewport(vertices, indices, triangles, viewport):
    # remove triangles that are entirely outside one of the viewport edges
    viewport = np.array(viewport)
    verts = vertices[indices[triangles]]
    cond = np.all([verts[:, i, 0] < viewport[0, 0] for i in range(3)], axis=0)
    cond = np.logical_or(cond, np.all([verts[:, i, 1] < viewport[0, 1] for i in range(3)], axis=0))
    cond = np.logical_or(cond, np.all([verts[:, i, 0] > viewport[1, 1] for i in range(3)], axis=0))
    cond = np.logical_or(cond, np.all([verts[:, i, 1] > viewport[1, 1] for i in range(3)], axis=0))
    return triangles[np.logical_not(cond)]


def to_homogeneous(vertices):
    vertices = np.array(vertices)
    if vertices.shape[1] < 4:
        vertices = np.concatenate((vertices, np.ones((vertices.shape[0], 4 - vertices.shape[1]))), axis=-1)
    return vertices


def to_screen_space(vertices, width, height):
    return (vertices * 0.5 + 0.5) * np.array([width, height, 0.5, 1.0], dtype=np.float32)\
           + np.array([[0, 0, 0.5, 0]], dtype=np.float32)


def block_triangles(vertices, indices, width, height, block_size=(16, 16)):
    """
    Create a matrix with triangle indices intersecting to blocks.
    :param vertices: triangle vertices in clip space.
    :param indices: triangle vertex indices.
    :param width: window width in pixels.
    :param height: window height in pixels.
    :param block_size: size of the block for rasterization in pixels.
    :return:
    """
    width_ex = block_size[0]
    while width_ex < width:
        width_ex *= 2
    height_ex = block_size[1]
    while height_ex < height:
        height_ex *= 2
    viewport = np.array([[0, 0], [width_ex, height_ex]])

    # transform vertices to screen space
    vertices = to_homogeneous(vertices)
    vertices = to_screen_space(vertices, width, height)

    # clip triangles
    triangles = triangles_in_viewport(vertices, np.array(indices), np.arange(len(indices)), viewport)

    # recursively distribute triangles in blocks
    triangles = split_triangles(vertices=vertices,
                                indices=np.array(indices),
                                triangles=triangles,
                                viewport=viewport,
                                block_size=block_size)

    # create block-triangle matrix
    num_triangles = len(indices)
    nx, ny = int(np.ceil(width / block_size[0])), int(np.ceil(height / block_size[1]))
    max_len = np.max([len(b) for v, b in triangles])
    block_triangles_mat = np.full((ny*nx, max_len), num_triangles)
    for v, b in triangles:
        if v[0, 0] < width and v[0, 1] < height:
            i = (v[0, 1] // block_size[1]) * nx + v[0, 0] // block_size[0]
            block_triangles_mat[i, :len(b)] = b

    return block_triangles_mat


def rasterize(attributes, indices, mvp, width, height, block_size=(16, 16)):
    """
    Rasterize triangles to create image.
    """
    assert width % block_size[0] == 0 and height % block_size[1] == 0

    # apply model-view-projection transform
    vertices = to_homogeneous(attributes['position'])
    vertices = vertices.T
    vertices = np.dot(mvp, vertices)
    vertices /= vertices[3]
    vertices = vertices.T

    # find block triangles
    block_triangles_mat = block_triangles(vertices, indices, width, height, block_size=block_size).transpose()

    # add one fake triangle
    num_vertices = len(vertices)
    attributes_exp = {}
    for name, attr in attributes.items():
        attributes_exp[name] = np.vstack((np.array(attr), [-1.0, -1.0, -1.0]))
    vertices = np.vstack((vertices, [2.0, 2.0, 2.0, 1.0]))
    indices = np.vstack((np.array(indices), [num_vertices, num_vertices, num_vertices]))

    # transform vertices to screen space
    vertices = to_screen_space(vertices, width, height)

    # rasterize triangles in blocks
    color_buffer = np.zeros((height, width, 3))
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    indices = np.array(indices)
    for i in range(block_triangles_mat.shape[0]):
        block_rasterize(color_buffer, depth_buffer, vertices, attributes_exp, indices, block_triangles_mat[i], width, height, block_size)

    # TODO: address y orientation
    return color_buffer[::-1, :, :]
