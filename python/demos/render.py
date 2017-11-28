import argparse
import cv2
from deep_slam.utils.np_raster import rasterize
from deep_slam.utils.transform import translation, rotation
from deep_slam.utils.projection import perspective
from deep_slam.utils.cube import cube
import numpy as np
import pyassimp
import time
import tkinter as tk


class App(object):
    def __init__(self, mesh, width=800, height=600, znear=0.1, zfar=10000.0):
        self.window_width = width
        self.window_height = height

        # Load mesh
        if mesh == 'cube':
            self.vertices, self.indices = cube()
            self.colors = [[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1],
                           [1, 1, 0],
                           [0, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1],
                           [0, 0, 0]]
        else:
            mesh = pyassimp.load(mesh)
            self.vertices = mesh.meshes[0].vertices
            self.normals = mesh.meshes[0].normals
            self.uvs = mesh.meshes[0].texturecoords[0, :, 1::-1]
            self.indices = mesh.meshes[0].faces
            self.colors = np.random.rand(*self.vertices.shape)

        # move scene back a little
        mean = np.mean(self.vertices, axis=0)
        stddev = np.std(self.vertices, axis=0)
        self.view = translation(-mean + [0, 0, stddev[2] * 8])
        self.proj = perspective(np.radians(90.0), width / height, znear, zfar)

        self.start_time = 0.0
        self.last_time = 0.0

    def run(self):
        cv2.namedWindow("render", cv2.WINDOW_NORMAL)
        cv2.moveWindow("render", self.window_width // 4, self.window_height // 4)
        cv2.resizeWindow('render', self.window_width, self.window_height)

        self.start_time = self.last_time = time.time()
        while True:
            theta = 0.03 * (time.time() - self.start_time)
            view = np.dot(self.view, rotation(theta, 0, 1, 0))
            mvp = np.dot(self.proj, view)

            attributes = {'position': self.vertices}
            if self.colors is not None:
                attributes['color'] = self.colors
            image = rasterize(attributes, self.indices, mvp=mvp, width=self.window_width, height=self.window_height)
            image = image[::-1, :, :]

            cv2.imshow("render", image)
            key = cv2.waitKey(1)
            if key == 27:
                return

            # Display frames per second
            cur_time = time.time()
            dtime = cur_time - self.last_time
            fps = 1.0 / float(dtime + 1e-9)
            self.last_time = cur_time
            cv2.setWindowTitle("render", "Renderer demo, fps = %.2f" % fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='renderer demo')
    parser.add_argument('mesh', nargs='?', default='cube', help='mesh to load or sample "cube" mesh')
    parser.add_argument('--width', default=None, help='window width')
    parser.add_argument('--height', default=None, help='window height')
    args = parser.parse_args()

    # autodetect screen resolution
    if args.width is None or args.height is None:
        root = tk.Tk()
        # determine window size, adjusted to produce 800x450 on a full-hd monitor
        if args.width is None:
            args.width = int(root.winfo_screenwidth() / 2.4) // 16 * 16
        if args.height is None:
            args.height = int(root.winfo_screenheight() / 2.4) // 16 * 16

    app = App(mesh=args.mesh, width=args.width, height=args.height)
    app.run()
