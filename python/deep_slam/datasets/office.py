import urllib.request
from zipfile import ZipFile
import os
from pathlib import Path
from deep_slam.datasets.VideoDataset import VideoDataset

home = str(Path.home())
data_dir = os.path.join(home, ".keras/datasets")

def report(count, blocksize, filesize):
    if count % 100 == 0:
        print("%d/%d" % (count*blocksize, filesize))

def load_data():
    url = "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/office.zip"
    if os.path.exists(os.path.join(data_dir, "office")):
        return VideoDataset(os.path.join(data_dir, "office/seq-01.zip"))
    elif not os.path.isfile(os.path.join(data_dir, "office.zip")):
        print("File downloading: %s " % os.path.join(data_dir, "office.zip"))
        file_name, headers = urllib.request.urlretrieve(url, os.path.join(data_dir, "office.zip"), report)
    else:
        file_name = os.path.join(data_dir, "office.zip")

    archive = ZipFile(file_name)
    "Extract seq-01.zip"
    archive.extractall(data_dir, ['office/seq-01.zip'])
    return VideoDataset(os.path.join(data_dir, "office/seq-01.zip"))

if __name__ == "__main__":
    load_data()
