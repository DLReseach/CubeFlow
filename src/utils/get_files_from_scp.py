import os
import sys
import tarfile
from pathlib import Path
from paramiko import SSHClient
import pickle
from scp import SCPClient
from src.utils.utils import get_project_root

remote_path = Path('/groups/hep/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/')
local_path = Path('/home/ehrhorn/data/CubeData/oscnext-genie-level5-v01-01-pass2')

done_dirs = [
    directory.name for directory in local_path.iterdir() if directory.is_dir()
]

ssh = SSHClient()
ssh.load_system_host_keys()
ssh.connect('hep01.hpc.ku.dk')

dirs = []

sftp = ssh.open_sftp()
for directory in sftp.listdir(path=str(remote_path)):
    dirs.append(directory)
sftp.close()

dirs = sorted(dirs)

for i, directory in enumerate(dirs):
    if directory not in done_dirs:
        print('Getting dir', directory)
        # current_local_path = local_path.joinpath(directory)
        # current_local_path.mkdir(exist_ok=False)
        current_remote_path = remote_path.joinpath(directory)
        command = 'rsync -zh --stats hep01:' + str(current_remote_path) + ' ' + str(local_path) + '/' + directory
        try:
            os.system(command)
            tar = tarfile.open(str(local_path) + '/' + directory)
            tar.extractall()
            tar.close()
            os.remove(str(local_path) + '/' + directory + '.tar')
            done_dirs.append(directory)
        except Exception as e:
            print(e)
    break
        

print('Done!')
