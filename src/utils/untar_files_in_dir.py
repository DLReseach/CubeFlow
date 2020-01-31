import os
import sys
import tarfile
import shutil
from pathlib import Path
from paramiko import SSHClient
import pickle
from scp import SCPClient
from src.utils.utils import get_project_root
from multiprocessing import Pool

TAR_PATH = Path.home().joinpath('data/CubeData/oscnext-genie-level5-v01-01-pass2/tarballs')
SAVE_PATH = Path.home().joinpath('data/CubeData/oscnext-genie-level5-v01-01-pass2/')
TAR_FILES = sorted([
    file for file in TAR_PATH.iterdir() if file.is_file() and file.suffix == '.tar'
])
tar_dir = 'lustre/hpc/hep/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/pickles/'


def untar_file(file):
    print('Dir:', file)
    tar = tarfile.open(file)
    tar.extractall(path=TAR_PATH)
    tar.close()
    tar_path = TAR_PATH.joinpath(tar_dir + '/' + file.stem)
    tar_path.rename(SAVE_PATH.joinpath(file.stem))
    os.remove(file)


if __name__ == '__main__':
    with Pool(48) as pool: 
        pool.map(untar_file, TAR_FILES)
    shutil.rmtree(TAR_PATH.joinpath('lustre'))
    print('Done!')
