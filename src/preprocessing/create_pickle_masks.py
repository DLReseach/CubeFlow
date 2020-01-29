from pathlib import Path
import pickle
from utils.utils import get_project_root

DATA_ROOT = get_project_root().joinpath(
    'data/toy_oscnext-genie-level5-v01-01-pass2'
)
MASK_ROOT = get_project_root().joinpath(
    'masks/toy_oscnext-genie-level5-v01-01-pass2'
)
DATA_DIRS = [
    directory for directory in DATA_ROOT.iterdir() if directory.is_dir()
]
PARTICLE_CODES = ['120000', '140000', '160000']


def get_event_particle_code(directory, mask_particle_code):
    mask = []
    files = [file for file in directory.iterdir() if file.is_file()]
    for file in files:
        with open(file, 'rb') as f:
            event_dict = pickle.load(f)
            event_particle_code = event_dict['meta']['particle_code']
            if event_particle_code == mask_particle_code:
                mask.append(file.stem)
    return mask


def create_particle_masks(directory, mask_particle_codes):
    particle_mask = {}
    for mask_particle_code in mask_particle_codes:
        particle_mask[mask_particle_code] = []
        for data_dir in directory:
            indices = get_event_particle_code(data_dir, mask_particle_code)
            particle_mask[mask_particle_code].extend(indices)
    return particle_mask


particle_mask = create_particle_masks(DATA_DIRS, PARTICLE_CODES)
PARTICLE_MASK_FILE = MASK_ROOT.joinpath('particle_codes.pickle')
with open(PARTICLE_MASK_FILE, 'wb') as f:
    pickle.dump(particle_mask, f)
