import argparse
import sys
import os
import os.path
from shutil import copyfile
from subprocess import call

temp_img_dir = "tmp"


def find_corresponding_frames(input_dir):
    f = open(os.path.join(input_dir, "INDEX.txt"))
    index = []
    next_rgb = ''
    next_depth = ''
    for line in f:
        fields = line.split('-')
        type = fields[0]
        timestamp = fields[2].split('.')[0]
        if type == 'r':
            next_rgb = line.rstrip()
        elif type == 'd':
            next_depth = line.rstrip()
        if next_rgb and next_depth:
            index.append([next_rgb, next_depth])
    return index


def move_files(input_dir, out_dir, index):
    os.makedirs(os.path.join(out_dir, 'rgb'))
    os.makedirs(os.path.join(out_dir, 'depth'))
    index_file = open(os.path.join(out_dir, 'index.txt'))
    for rgb_file, depth_file in index:
        copyfile(os.path.join(input_dir, rgb_file), os.path.join(out_dir, 'rgb', rgb_file))
        copyfile(os.path.join(input_dir, depth_file), os.path.join(out_dir, 'depth', depth_file))
        index_file.write("%s %s" % (os.path.join('rgb', rgb_file), os.path.join('depth', depth_file)))


def run_kinect(fakenect_record_path, temp_out):
    call([fakenect_record_path, temp_out])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data from kinect')
    parser.add_argument('fakenect_record', help='fakenect-record binary path')
    parser.add_argument('out', help='output directory')
    args = parser.parse_args()
    run_kinect(args.fakenect_record, temp_img_dir)
