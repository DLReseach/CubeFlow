from pathlib import Path
import pickle
from multiprocessing import Pool, cpu_count
from multiprocessing import Process, Manager
from utils.utils import get_project_root

DATA_ROOT = get_project_root().joinpath(
    'data/oscnext-genie-level5-v01-01-pass2'
)
MASK_ROOT = get_project_root().joinpath(
    'masks/oscnext-genie-level5-v01-01-pass2'
)
MASK_ROOT.mkdir(exist_ok=True)

DATA_DIRS = sorted([
    directory for directory in DATA_ROOT.iterdir() if directory.is_dir()
])
DATA_DIRS = DATA_DIRS[0:4]
print(DATA_DIRS)
PARTICLE_CODES = ['120000', '140000', '160000']
MAX_DOMS = [200]


def get_event_particle_code(directory, mask_particle_code, mask_list):
    mask = []
    print(directory)
    files = [file for file in directory.iterdir() if file.is_file()]
    for i, file in enumerate(files):
        if i % 100 == 0:
            print(i)
        with open(file, 'rb') as f:
            event_dict = pickle.load(f)
            event_particle_code = event_dict['meta']['particle_code']
            if event_particle_code == mask_particle_code:
                mask.append(file.stem)
    mask_list.extend(mask)


def get_event_length(directory, max_doms, mask_list):
    mask = []
    print(directory)
    files = [file for file in directory.iterdir() if file.is_file()]
    for i, file in enumerate(files):
        if i % 100 == 0:
            print(i)
        with open(file, 'rb') as f:
            event_dict = pickle.load(f)
            if len(event_dict['masks']['SplitInIcePulses']) <= 200:
                mask.append(file.stem)
    mask_list.extend(mask)


particle_dict = {}
for particle_code in PARTICLE_CODES:
    print('Particle code mask')
    mask_list = []
    # with Manager() as manager:
    #     mask_list = manager.list()
    #     processes = []
    #     for i in range(4):
    #         p = Process(
    #             target=get_event_particle_code,
    #             args=(DATA_DIRS[i], particle_code, mask_list)
    #         )
    #         p.start()
    #         processes.append(p)
    #     for p in processes:
    #         p.join()
    for data_dir in DATA_DIRS:
        get_event_particle_code(data_dir, particle_code, mask_list)
    particle_dict[particle_code] = mask_list
PARTICLE_CODES_FILE = MASK_ROOT.joinpath('particle_codes.pickle')
with open(PARTICLE_CODES_FILE, 'wb') as f:
    pickle.dump(particle_dict, f)
max_doms_dict = {}
for max_doms in MAX_DOMS:
    print('Max doms mask')
    mask_list = []
    for data_dir in DATA_DIRS:
        get_event_length(data_dir, max_doms, mask_list)
    # with Manager() as manager:
    #     mask_list = manager.list()
    #     processes = []
    #     for i in range(4):
    #         p = Process(
    #             target=get_event_length,
    #             args=(DATA_DIRS[i], max_doms, mask_list)
    #         )
    #         p.start()
    #         processes.append(p)
    #     for p in processes:
    #         p.join()
    max_doms_dict[max_doms] = mask_list
MAX_DOMS_FILE = MASK_ROOT.joinpath('max_doms.pickle')
with open(MAX_DOMS_FILE, 'wb') as f:
    pickle.dump(max_doms_dict, f)
