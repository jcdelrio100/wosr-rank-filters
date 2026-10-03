"""Download the public datasets used by the benchmarks (not redistributed in this repository).

  code/data/img/Set12, code/data/img/BSD68   test images      (from the DnCNN repository, K. Zhang et al.)
  code/data/img/Train400                     training images  (from the DnCNN repository)
  code/data/fmnist/*.gz                      Fashion-MNIST    (Zalando Research)

Usage:  python code/data/get_data.py        (needs git and an internet connection; ~50 MB)
"""
import os, shutil, subprocess, tempfile, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DNCNN = "https://github.com/cszn/DnCNN.git"
SETS = {"Set12": "testsets/Set12", "BSD68": "testsets/BSD68", "Train400": "TrainingCodes/DnCNN_TrainingCodes_v1.0/data/Train400"}
FMNIST = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FFILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]

def images():
    dst = os.path.join(HERE, "img")
    if all(os.path.isdir(os.path.join(dst, k)) for k in SETS):
        print("images already present"); return
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", DNCNN, tmp], check=True)
        subprocess.run(["git", "-C", tmp, "sparse-checkout", "set", *SETS.values()], check=True)
        for k, p in SETS.items():
            shutil.copytree(os.path.join(tmp, p), os.path.join(dst, k), dirs_exist_ok=True)
            print(k, len(os.listdir(os.path.join(dst, k))), "files")

def fmnist():
    dst = os.path.join(HERE, "fmnist"); os.makedirs(dst, exist_ok=True)
    for f in FFILES:
        out = os.path.join(dst, f)
        if not os.path.exists(out):
            urllib.request.urlretrieve(FMNIST + f, out); print("downloaded", f)

if __name__ == "__main__":
    images(); fmnist()
