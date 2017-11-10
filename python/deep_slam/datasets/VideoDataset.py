
import numpy as np
import cv2
import os
import argparse
from zipfile import ZipFile
import io




class VideoDataset(object):
    def __init__(self, path, color_mode='rgbd', target_size=None):
        """

        :param path: dataset folder
        :param color_mode: rgb or rgbd
        :param target_size:
        """
        self.path = path
        self.color_mode = color_mode
        self.target_size = target_size
        self.compressed = False
        if os.path.splitext(path)[1] == '.zip':
            self.compressed = True
            self.archive = ZipFile(path, 'a')
            self.filenames = self.archive.namelist()
        else:
            self.filenames = os.listdir(path)

        if color_mode == 'rgbd':
            self.num_channels = 3
            self.num_samples = len(self.filenames) // 3
        elif color_mode == 'rgb':
            self.num_channels = 3
            self.num_samples = len(self.filenames) // 2

        self.rgb_template = "frame-%06d.color.png"
        self.depth_template = "frame-%06d.depth.png"
        self.pose_template = "frame-%06d.pose.txt"
        if self.compressed:
            archive_name = os.path.splitext(os.path.basename(path))[0]
            self.rgb_template = "%s/%s" % (archive_name, self.rgb_template)
            self.depth_template = "%s/%s" % (archive_name, self.depth_template)
            self.pose_template = "%s/%s" % (archive_name, self.pose_template)
            self.path = ""
        if not target_size:
            rgb_file = os.path.join(self.path, self.rgb_template % (0,))
            rgb_img = self.imread(rgb_file, cv2.IMREAD_COLOR)
            self.target_size = [rgb_img.shape[0], rgb_img.shape[1]]

    def imread(self, fname, flags):
        if self.compressed:
            data = self.archive.read(fname)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), flags)
        else:
            img = cv2.imread(fname, cv2.IMREAD_COLOR)
        return img

    def imwrite(self, fname, img):
        if self.compressed:
            _, data = cv2.imencode('.png', img)
            self.archive.writestr(fname, data)
        else:
            cv2.imwrite(fname, img)

    def loadtxt(self, filename):
        if self.compressed:
            filename = self.archive.open(filename)
        return np.loadtxt(filename)

    def savetxt(self, filename, mat):
        if self.compressed:
            f = io.BytesIO()
            np.savetxt(f, mat)
            self.archive.writestr(filename, f.getvalue())
        else:
            np.savetxt(filename, mat)


    def get_batch(self, offset, batchsize=1):
        img_batch = np.zeros([batchsize, self.target_size[0], self.target_size[1], self.num_channels])
        depth_batch = np.zeros([batchsize, self.target_size[0], self.target_size[1]])
        pose_batch = np.zeros([batchsize, 4, 4])

        for idx in range(batchsize):
            rgb_file = os.path.join(self.path, self.rgb_template % ((offset + idx) % self.num_samples,))
            pose_file = os.path.join(self.path, self.pose_template % ((offset + idx) % self.num_samples,))

            rgb_img = self.imread(rgb_file, cv2.IMREAD_COLOR)
            if self.color_mode == 'rgbd':
                depth_file = os.path.join(self.path, self.depth_template % (offset,))
                depth_img = self.imread(depth_file, cv2.IMREAD_GRAYSCALE)

                if depth_img.shape[0] != self.target_size[0] or depth_img.shape[1] != self.target_size[1]:
                    depth_img = cv2.resize(depth_img, (self.target_size[1], self.target_size[0]),
                                           interpolation=cv2.INTER_AREA)
                depth_batch[idx, :, :] = depth_img
            pose = self.loadtxt(pose_file)

            if rgb_img.shape[0] != self.target_size[0] or rgb_img.shape[1] != self.target_size[1]:
                rgb_img = cv2.resize(rgb_img, (self.target_size[1], self.target_size[0]),
                                       interpolation=cv2.INTER_AREA)

            img_batch[idx, :, :, :] = rgb_img
            pose_batch[idx, :, :] = pose
        if self.color_mode == 'rgbd':
            return img_batch, depth_batch, pose_batch
        else:
            return img_batch, pose_batch

    def generate_batches(self, batchsize=1):
        for offset in range(0, self.num_samples, batchsize):
            yield self.get_batch(offset, batchsize)

    def add_sample(self, *args):
        if self.color_mode == 'rgbd':
            img, depth, pose = args
        else:
            img, pose = args

        rgb_file = os.path.join(self.path, self.rgb_template % (self.num_samples,))
        pose_file = os.path.join(self.path, self.pose_template % (self.num_samples,))
        self.imwrite(rgb_file, img)
        self.savetxt(pose_file, pose)
        if self.color_mode == 'rgbd':
            depth_file = os.path.join(self.path, self.depth_template % (self.num_samples,))
            self.imwrite(depth_file, depth)
        self.num_samples += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    args = parser.parse_args()
    dataset = VideoDataset(args.dataset)
    imgs, depth, poses = dataset.get_batch(0, 1)
    dataset.add_sample(imgs[0], depth[0], poses[0])
    print(imgs.shape, depth.shape, poses.shape)
    print(poses)