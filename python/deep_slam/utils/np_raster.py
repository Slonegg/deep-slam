import numpy as np


def barycentric(p, t):
    """
    For each point p find its barycentric coordinates s, t in triangle t.
    :param p: array of points of shape (height, width, 2)
    :param t: array of triangle vertices of shape (height, width, 3, 2..4)
    :return: boolean array of shape (height, width), entry is true if point p[i, j] is in triangle t[i, j]
    """
    na = np.newaxis

    A = -t[1, 1] * t[0, 2] + t[1, 0] * (-t[0, 1] + t[0, 2])\
        + t[0, 0] * (t[1, 1] - t[1, 2]) + t[0, 1] * t[1, 2]
    u = (t[1, 0] * t[0, 2] - t[0, 0] * t[1, 2])[:, na, na]\
        + (t[1, 2] - t[1, 0])[:, na, na] * p[0]\
        + (t[0, 0] - t[0, 2])[:, na, na] * p[1]
    u *= np.sign(A)[:, na, na]
    v = (t[0, 0] * t[1, 1] - t[1, 0] * t[0, 1])[:, na, na]\
        + (t[1, 0] - t[1, 1])[:, na, na] * p[0]\
        + (t[0, 1] - t[0, 0])[:, na, na] * p[1]
    v *= np.sign(A)[:, na, na]

    mask = np.logical_and(u >= 0, v >= 0)
    mask = np.logical_and(mask, (u + v) < np.abs(A)[:, na, na])

    A = np.maximum(np.abs(A), np.finfo(np.float32).eps)
    u /= A[:, na, na]
    v /= A[:, na, na]

    return u, v, mask


def persp_interp(bar, vertex_inv_z, pixel_inv_z, vertex_attr):
    """
    Perform perspective corrected interpolation in screen space.
    See [here](interpolation https://www.comp.nus.edu.sg/~lowkl/publications/lowk_persp_interp_techrep.pdf)
    :param u: array of barycentric coordinates corresponding first triangle vertex of shape (height, width)
    :param v: array of barycentric coordinates corresponding second triangle vertex of shape (height, width)
    :param z: array of triangle vertex z coordinates of shape (height, width, 3)
    :param x: array of interpolant values at triangle vertices of shape (height, width, n)
    :return: array of interpolant values at pixels of shape (height, width, n)
    """
    na = np.newaxis
    if vertex_attr.ndim == 3:
        return np.sum(bar[na, :] * vertex_attr[:, :, :, na, na] * vertex_inv_z[na, :, :, na, na], axis=1) / pixel_inv_z[na]
    elif vertex_attr.ndim == 2:
        return np.sum(bar * vertex_attr[:, :, na, na] * vertex_inv_z[:, :, na, na], axis=0) / pixel_inv_z
    else:
        raise RuntimeError('Unexpected number of dimensions of attribute array')


def block_rasterize(vertices, attributes, indices, block_inds, block_tris, width, height, block_size):
    # coordinates of the triangles in the block, (height, width, 3, xy)
    nx, ny = width // block_size[0], height // block_size[1]
    x = np.zeros((nx*ny, block_size[0]), dtype=np.int32)
    y = np.zeros((nx*ny, block_size[1]), dtype=np.int32)
    for i in range(ny):
        for j in range(nx):
            x[i*nx + j] = np.arange(j*block_size[0], (j+1)*block_size[0]) + 0.5
            y[i*nx + j] = np.arange(i*block_size[1], (i+1)*block_size[1]) + 0.5
    triangles = vertices[:, indices[:, block_tris]]

    # find barycentric coordinates
    points = np.array([np.meshgrid(x[i], y[i]) for i in block_inds])
    points = np.moveaxis(points, 0, 1)
    s, t, mask = barycentric(points, triangles)
    bar = np.stack((1 - s - t, s, t))

    # perform z-test
    vertex_inv_z = 1 / np.maximum(triangles[2], np.finfo(np.float32).eps)
    pixel_inv_z = np.sum(bar * vertex_inv_z[:, :, np.newaxis, np.newaxis], axis=0)
    pixel_inv_z[np.logical_not(mask)] = np.inf

    # run pixel shader
    varyings = {}
    for name, attribute in attributes.items():
        if name not in ('position',):
            x = attribute[:, indices[:, block_tris]]
            varyings[name] = persp_interp(bar, vertex_inv_z, pixel_inv_z, x)
    color = varyings['color'] if 'color' in varyings else 255
    color = np.rollaxis(color, 0, 4)

    return {'color': color, 'depth': pixel_inv_z, 'rmask': mask}


def block_merge(block_inds, block_rt, width, height, block_size):
    color = block_rt['color']
    depth = block_rt['depth']
    rmask = block_rt['rmask']

    color_buffer = np.zeros((height, width, 3), dtype=np.float32)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)

    nx, ny = width // block_size[0], height // block_size[1]
    for i in range(ny):
        for j in range(nx):
            block_mask = np.where(block_inds == i*nx + j)[0]
            if len(block_mask) == 0:
                continue

            def select(A, idx):
                m, n = A.shape[1], A.shape[2]
                I, J = np.ogrid[:m, :n]
                return A[idx, I, J]

            depth_argmin = np.argmin(depth[block_mask], axis=0)
            block_color = select(color[block_mask], depth_argmin)
            block_depth = select(depth[block_mask], depth_argmin)
            block_rmask = select(rmask[block_mask], depth_argmin)

            oy = i*block_size[1]
            ox = j*block_size[0]
            color_buffer[oy:oy+block_size[1], ox:ox+block_size[0]][block_rmask] = block_color[block_rmask]
            depth_buffer[oy:oy+block_size[1], ox:ox+block_size[0]][block_rmask] = block_depth[block_rmask]

    return {'color': color_buffer, 'depth': depth_buffer}


def split_viewport(vp, axis, split):
    if axis == 0:
        v0 = (vp[0][0], vp[0][1]), (split, vp[1][1])
        v1 = (split, vp[0][1]), (vp[1][0], vp[1][1])
    else:
        v0 = (vp[0][0], vp[0][1]), (vp[1][0], split)
        v1 = (vp[0][0], split), (vp[1][0], vp[1][1])
    return v0, v1


def split_triangles(vertices, indices, triangles, viewport, block_size, width, height):
    viewport_width = viewport[1][0] - viewport[0][0]
    viewport_height = viewport[1][1] - viewport[0][1]
    if viewport_width == block_size[0] and viewport_height == block_size[1]:
        return [(viewport, triangles)]

    # split viewport
    axis = 0 if viewport_width > viewport_height else 1
    split = (viewport[0][axis] + viewport[1][axis]) // 2
    if split >= (width, height)[axis]:
        return []
    v0, v1 = split_viewport(viewport, axis, split)

    # split triangles
    v = vertices[:, indices[:, triangles]]
    tri_l = triangles[np.logical_or(v[axis, 0] < split, np.logical_or(v[axis, 1] < split, v[axis, 2] < split))]
    tri_r = triangles[np.logical_or(v[axis, 0] >= split, np.logical_or(v[axis, 1] >= split, v[axis, 2] >= split))]

    return split_triangles(vertices, indices, tri_l, v0, block_size, width, height)\
        + split_triangles(vertices, indices, tri_r, v1, block_size, width, height)


def triangles_in_viewport(vertices, indices, triangles, viewport):
    # remove triangles that are entirely outside one of the viewport edges
    viewport = np.array(viewport)
    v = vertices[:, indices[:, triangles]]
    cond = np.all([v[0, i] < viewport[0, 0] for i in range(3)], axis=0)
    cond = np.logical_or(cond, np.all([v[1, i] < viewport[0, 1] for i in range(3)], axis=0))
    cond = np.logical_or(cond, np.all([v[0, i] > viewport[1, 1] for i in range(3)], axis=0))
    cond = np.logical_or(cond, np.all([v[1, i] > viewport[1, 1] for i in range(3)], axis=0))
    return triangles[np.logical_not(cond)]


def to_homogeneous(vertices):
    if vertices.shape[0] < 4:
        vertices = np.concatenate((vertices, np.ones((4 - vertices.shape[0], *vertices.shape[1:]))), axis=0)
    return vertices


def to_screen_space(vertices, width, height):
    vertices = vertices * 0.5 + 0.5
    vertices[0] *= width
    vertices[1] *= height
    vertices[2] = vertices[2] * 0.5 + 0.5
    return vertices


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
    viewport = ((0, 0), (width_ex, height_ex))

    # transform vertices to screen space
    vertices = to_homogeneous(vertices)
    vertices = to_screen_space(vertices, width, height)

    # clip triangles
    indices = np.array(indices)
    triangles = triangles_in_viewport(vertices, indices, np.arange(indices.shape[1]), viewport)

    # recursively distribute triangles in blocks
    triangles = split_triangles(vertices=vertices,
                                indices=np.array(indices),
                                triangles=triangles,
                                viewport=viewport,
                                block_size=block_size,
                                width=width,
                                height=height)

    # create block-triangle matrix
    nx, ny = int(np.ceil(width / block_size[0])), int(np.ceil(height / block_size[1]))
    block_inds = []
    block_tris = []
    for v, b in triangles:
        i = (v[0][1] // block_size[1]) * nx + v[0][0] // block_size[0]
        block_inds += [i] * len(b)
        block_tris += b.tolist()

    return np.array(block_inds), np.array(block_tris)


def rasterize(attributes, indices, mvp, width, height, block_size=(16, 16)):
    """
    Rasterize triangles to create image.
    """
    assert width % block_size[0] == 0 and height % block_size[1] == 0

    # make numpy arrays and transpose attributes, it is much easier to work with them this way
    for name, attribute in attributes.items():
        attributes[name] = np.array(attribute).T
    indices = np.array(indices).T

    # apply model-view-projection transform
    vertices = to_homogeneous(attributes['position'])
    vertices = np.dot(mvp, vertices)
    vertices /= vertices[3]

    # find block triangles
    block_inds, block_tris = block_triangles(vertices, indices, width, height, block_size=block_size)

    # transform vertices to screen space
    vertices = to_screen_space(vertices, width, height)

    # rasterize triangles in blocks
    block_rt = block_rasterize(vertices, attributes, indices, block_inds, block_tris, width, height, block_size)
    rt = block_merge(block_inds, block_rt, width, height, block_size)

    # TODO: address y orientation
    return rt['color'][::-1, :, :]
