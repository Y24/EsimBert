import sys
sys.path.append("/home/y24/毕设/EsimBert/")
import os
from util.path import get_root_path
import wget
import argparse
import zipfile
from typing import Any, List, Set, Text, Tuple, Union


def download(url: str, targetdir: str):
    print("Downloading from {} ...".format(url))
    filepath = os.path.join(targetdir, url.split('/')[-1])
    wget.download(url, filepath)
    return filepath


def contains(container: Union[Tuple, List, Set], target: Any):
    for item in container:
        if(item in target):
            return True
    return False


def unzip(filepath: str, igorned: Set[Text]):
    print("Extracting: {} ...".format(filepath))
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        for name in zf.namelist():
            if contains(igorned, name):
                continue
            zf.extract(name, dirpath)


def exists(target: str):
    return os.path.exists(target)


def fetch_data(url: str, targetdir: str, ignored: Set[Text]):
    filepath = os.path.join(targetdir, url.split('/')[-1])
    target = os.path.join(targetdir,
                          ".".join((url.split('/')[-1]).split('.')[:-1]))
    if not exists(targetdir):
        print("Creating target directory {}...".format(targetdir))
        os.makedirs(targetdir)
    if exists(target) or exists(target+".txt"):
        print("Found unzipped data in {}, skipping download and unzip..."
              .format(targetdir))
    elif exists(filepath):
        print("Found zipped data in {} - skipping download..."
              .format(targetdir))
        unzip(filepath, igorned=ignored)
        os.remove(filepath)
    else:
        resultpath = download(url, targetdir)
        unzip(resultpath, igorned=ignored)
        os.remove(resultpath)



if __name__ == "__main__":
    snli_url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
    snli_ignored: Set[Text] = ("__MACOSX", "Icon", ".DS_Store")
    parser = argparse.ArgumentParser(description="Download snli_1.0 dataset.")
    parser.add_argument("--dataset_url",
                        default=snli_url,
                        help="URL of the dataset to download")
    parser.add_argument("--target_dir",
                        default=os.path.join(
                            get_root_path(), "data/dataset"),
                        help="Path to a directory where data must be saved")
    args = parser.parse_args()
    fetch_data(args.dataset_url, args.target_dir, ignored=snli_ignored)
