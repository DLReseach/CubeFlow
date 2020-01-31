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

remote_path = Path('/groups/hep/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/tarballs')
local_path = Path('/home/ehrhorn/data/CubeData/oscnext-genie-level5-v01-01-pass2')
done_dirs_file = Path('/home/ehrhorn/repos/CubeFlow/done_dirs.pickle')
tar_dir = 'lustre/hpc/hep/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/pickles/'

done_dirs = [
    directory.name for directory in local_path.iterdir() if directory.is_dir()
]

ssh = SSHClient()
ssh.load_system_host_keys()
ssh.connect('hep01.hpc.ku.dk')

files = []

sftp = ssh.open_sftp()
for file in sftp.listdir(path=str(remote_path)):
    files.append(file)
sftp.close()

files = sorted(files)


def get_files(file):
    if file.split('.')[0] not in done_dirs:
        print('Getting file', file)
        current_local_path = local_path.joinpath(file.split('.')[0])
        current_local_path.mkdir(exist_ok=False)
        current_remote_path = remote_path.joinpath(file)
        command = 'rsync -zh --stats hep01:' + str(current_remote_path) + ' ' + str(local_path)
        try:
            os.system(command)
            current_file = local_path.joinpath(file)
            tar = tarfile.open(current_file)
            tar.extractall(path=local_path)
            tar.close()
            tar_path = local_path.joinpath(tar_dir + '/' + file.split('.')[0])
            tar_path.rename(local_path.joinpath(file.split('.')[0]))
            os.remove(current_file)
        except Exception as e:
            shutil.rmtree(current_local_path)
            print(e)


if __name__ == '__main__':
    with Pool(12) as pool: 
        pool.map(get_files, files)
    shutil.rmtree(local_path.joinpath('lustre'))
    print('Done!')
