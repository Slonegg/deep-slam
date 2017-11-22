
import numpy as np
import cv2
import os


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
        if not target_size:
            rgb_file = os.path.join(self.path, self.rgb_template % (0,))
            rgb_img = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
            self.target_size = [rgb_img.shape[0], rgb_img.shape[1]]

    def imread(self, file, flag):
        img = cv2.imread(file, flag)
        if img.shape[0] != self.target_size[0] or img.shape[1] != self.target_size[1]:
            img = cv2.resize(img, (self.target_size[1], self.target_size[0]),
                             interpolation=cv2.INTER_AREA)
        return img

    def get_sample(self, offset):
        rgb_file = os.path.join(self.path, self.rgb_template % (offset % self.num_samples,))
        pose_file = os.path.join(self.path, self.pose_template % (offset % self.num_samples,))
        rgb_img = self.imread(rgb_file, cv2.IMREAD_COLOR)
        pose = np.loadtxt(pose_file)
        out = {'rgb': rgb_img, 'pose': pose}

        if self.color_mode == 'rgbd':
            depth_file = os.path.join(self.path, self.depth_template % (offset,))
            depth_img = self.imread(depth_file, cv2.IMREAD_ANYDEPTH)
            out['depth'] = depth_img
        return out

    def generator(self):
        for offset in range(0, self.num_samples):
            yield self.get_sample(offset)

    def add_sample(self, *args):
        if self.color_mode == 'rgbd':
            img, depth, pose = args
        else:
            img, pose = args

        rgb_file = os.path.join(self.path, self.rgb_template % (self.num_samples,))
        pose_file = os.path.join(self.path, self.pose_template % (self.num_samples,))
        cv2.imwrite(rgb_file, img)
        np.savetxt(pose_file, pose)
        if self.color_mode == 'rgbd':
            depth_file = os.path.join(self.path, self.depth_template % (self.num_samples,))
            cv2.imwrite(depth_file, depth)
        self.num_samples += 1
