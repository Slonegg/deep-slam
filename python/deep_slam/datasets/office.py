import urllib.request
from zipfile import ZipFile
import os
from deep_slam.datasets.VideoDataset import VideoDataset


def report(count, blocksize, filesize):
    if count % 100 == 0:
        print("%d/%d" % (count*blocksize, filesize))


def load_data(video_id=1, data_dir=None):
    if not data_dir:
        data_dir = os.path.expanduser('~/.keras/datasets')
    url = "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/office.zip"
    subdir = "seq-%02d" % video_id
    subarchive = "%s.zip" % subdir
    full_arhive_path = os.path.join(data_dir, "office.zip")
    sub_archive_path = os.path.join(data_dir, "office", subarchive)
    sub_dir_path = os.path.join(data_dir, "office", subdir)

    if not os.path.exists(sub_dir_path) and\
            not os.path.isfile(sub_archive_path) and \
            not os.path.isfile(full_arhive_path):
        # No data at all
        # Download data
        print("File downloading: %s " % full_arhive_path)
        full_arhive_path, headers = urllib.request.urlretrieve(url, full_arhive_path, report)
    if not os.path.exists(sub_dir_path) and\
            not os.path.isfile(sub_archive_path):
        # extract subarchive #video_id from full archive
        with ZipFile(full_arhive_path) as archive:
            archive.extractall(data_dir, ['office/' + subarchive])
    if not os.path.exists(sub_dir_path):
        # extract files from sub archive
        with ZipFile(sub_archive_path) as archive:
            archive.extractall(os.path.join(data_dir, 'office'))

    return VideoDataset(sub_dir_path)

if __name__ == "__main__":
    data = load_data()
    for i, f in enumerate(data.generator()):
        print(i)
        print(f['rgb'].shape)
        print(f['depth'].shape)
        print(f['pose'])
