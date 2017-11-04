
import numpy as np
import cv2
import os
import argparse


class VideoDataset(object):
    def __init__(self, path, color_mode='rgbd', target_size=None):
        """

        :param path: dataset folder
        :param color_mode: rgb or rgbd
        :param target_size:
        """
        self.path = path
        self.filenames = os.listdir(path)
        self.color_mode = color_mode
        self.target_size = target_size
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

    def get_batch(self, offset, batchsize=1):
        img_batch = np.zeros([batchsize, self.target_size[0], self.target_size[1], self.num_channels])
        depth_batch = np.zeros([batchsize, self.target_size[0], self.target_size[1]])
        pose_batch = np.zeros([batchsize, 4, 4])

        for idx in range(batchsize):
            rgb_file = os.path.join(self.path, self.rgb_template % ((offset + idx) % self.num_samples,))
            pose_file = os.path.join(self.path, self.pose_template % ((offset + idx) % self.num_samples,))

            rgb_img = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
            if self.color_mode == 'rgbd':
                depth_file = os.path.join(self.path, self.depth_template % (offset,))
                depth_img = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE)

                if depth_img.shape[0] != self.target_size[0] or depth_img.shape[1] != self.target_size[1]:
                    depth_img = cv2.resize(depth_img, (self.target_size[1], self.target_size[0]),
                                           interpolation=cv2.INTER_AREA)
                depth_batch[idx, :, :] = depth_img
            pose = np.loadtxt(pose_file)

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
        cv2.imwrite(rgb_file, img)
        np.savetxt(pose_file, pose)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    args = parser.parse_args()
    dataset = VideoDataset(args.dataset)
    imgs, depth, poses = dataset.get_batch(0, 1)
    print(imgs.shape, depth.shape, poses.shape)