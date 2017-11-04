import tensorflow as tf
from zipfile import ZipFile
import os
import sys
from scipy import misc
import numpy as np
import argparse

rgb_template = "seq-%02d/frame-%06d.color.png"
depth_template = "seq-%02d/frame-%06d.depth.png"
pose_template = "seq-%02d/frame-%06d.pose.txt"
FLAGS = None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(bytes_list=tf.train.FloatList(value=[value]))


def convert_to(data_set, seq_id, name):
    """Converts a dataset to tfrecords."""
    archive = ZipFile(os.path.join(data_set, "seq-%02d.zip" % seq_id))
    num_examples = len(archive.namelist()) // 3

    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(num_examples):
        rgb_file = rgb_template % (seq_id, index)
        depth_file = depth_template % (seq_id, index)
        pose_file = pose_template % (seq_id, index)
        with archive.open(rgb_file) as file:
            rgb_img = misc.imread(file)
        with archive.open(depth_file) as file:
            depth_img = misc.imread(file)
        img = np.dstack((rgb_img, depth_img))
        image_raw = img.tostring()
        rows = img.shape[0]
        cols = img.shape[1]
        with archive.open(pose_file) as file:
            pose = np.loadtxt(file)
            print(type(pose))
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'pose_raw': _bytes_feature(pose.tostring()),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument(
        '--directory',
        type=str,
        default='/tmp/data',
        help='Directory to download data files and write the converted result'
    )
    FLAGS, unparsed = parser.parse_known_args()
    convert_to(FLAGS.dataset, 1, "office-01")