import urllib.request
from zipfile import ZipFile
import os

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "data")


def load_office():
    url = ""
    file_name, headers = urllib.request.urlretrieve(url)
    archive = ZipFile(file_name)
    archive.extractall()