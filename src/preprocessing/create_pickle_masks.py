from pathlib import Path
import pickle
from multiprocessing import Pool, cpu_count
from multiprocessing import Process, Manager
from utils.utils import get_project_root

DATA_ROOT = Path.home().joinpath('small_data_test/oscnext-genie-level5-v01-01-pass2/pickles')
MASK_ROOT = Path.home().joinpath('small_data_test/oscnext-genie-level5-v01-01-pass2/masks')

MASK_ROOT.mkdir(exist_ok=True)

DATA_DIRS = sorted([
    directory for directory in DATA_ROOT.iterdir() if directory.is_dir()
    and directory.stem != 'masks' and directory.stem != 'transformers'
])
PARTICLE_CODES = ['140000']
MAX_DOMS = [200]
MAX_ENERGY = [3]


def get_event_particle_code(directory, mask_particle_code, mask_list):
    mask = []
    files = [file for file in directory.iterdir() if file.is_file()]
    for i, file in enumerate(files):
        with open(file, 'rb') as f:
            event_dict = pickle.load(f)
            event_particle_code = event_dict['meta']['particle_code']
            if event_particle_code == mask_particle_code:
                mask.append(file.stem)
    mask_list.extend(mask)


def get_event_length(directory, max_doms, mask_list):
    mask = []
    files = [file for file in directory.iterdir() if file.is_file()]
    for i, file in enumerate(files):
        with open(file, 'rb') as f:
            event_dict = pickle.load(f)
            if len(event_dict['masks']['SplitInIcePulses']) <= 200:
                mask.append(file.stem)
    mask_list.extend(mask)


def get_event_energy(directory, mex_energy, mask_list):
    mask = []
    files = [file for file in directory.iterdir() if file.is_file()]
    for i, file in enumerate(files):
        with open(file, 'rb') as f:
            event_dict = pickle.load(f)
            if event_dict['raw']['true_primary_energy'] <= 3:
                mask.append(file.stem)
    mask_list.extend(mask)


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
        print(data_dir)
        get_event_particle_code(data_dir, particle_code, mask_list)
PARTICLE_CODES_FILE = MASK_ROOT.joinpath('muon_neutrino.pickle')
with open(PARTICLE_CODES_FILE, 'wb') as f:
    pickle.dump(mask_list, f)
for max_doms in MAX_DOMS:
    print('Max doms mask')
    mask_list = []
    for data_dir in DATA_DIRS:
        print(data_dir)
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
MAX_DOMS_FILE = MASK_ROOT.joinpath('dom_interval_SplitInIcePulses_min0_max200.pickle')
with open(MAX_DOMS_FILE, 'wb') as f:
    pickle.dump(mask_list, f)
with open(PARTICLE_CODES_FILE, 'wb') as f:
    pickle.dump(mask_list, f)
for max_energy in MAX_ENERGY:
    print('Max energy mask')
    mask_list = []
    for data_dir in DATA_DIRS:
        print(data_dir)
        get_event_energy(data_dir, max_energy, mask_list)
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
MAX_ENERGY_FILE = MASK_ROOT.joinpath('energy_interval_min0.0_max3.0.pickle')
with open(MAX_ENERGY_FILE, 'wb') as f:
    pickle.dump(mask_list, f)

print('Done!')
