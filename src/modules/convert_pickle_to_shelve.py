import os
import psutil
import pickle
import shelve
from pathlib import Path
import datetime

from src.modules.utils import get_time

process = psutil.Process(os.getpid())

HOME_PATH = Path().home()

SHELVE_PATH = HOME_PATH.joinpath(
    'small_data_test/oscnext-genie-level5-v01-01-pass2/shelve/')
SHELVE_PATH.mkdir(exist_ok=True)
SHELVE_NAME = SHELVE_PATH.joinpath('cube_shelve')
SHELVE_DATA_FILE = Path(str(SHELVE_NAME) + '.dat')

DATA_PATH = HOME_PATH.joinpath(
    'small_data_test/oscnext-genie-level5-v01-01-pass2/pickles'
)
PICKLE_DIRS = [
    directory for directory in DATA_PATH.iterdir() if directory.is_dir()
]

shelve_file_exists = SHELVE_DATA_FILE.is_file()

if shelve_file_exists:
    with shelve.open(str(SHELVE_NAME), 'r') as f:
        EXISTING_EVENTS = list(f.keys())
else:
    EXISTING_EVENTS = []

for directory in PICKLE_DIRS:
    print(
        '{}: Handling directory {}'.format(
            get_time(),
            directory.stem
        )
    )

    time_start = datetime.datetime.now()

    files = [
        file for file in directory.glob('**/*') if file.suffix == '.pickle'
    ]

    with shelve.open(str(SHELVE_NAME), 'c') as db:
        for file in files:
            if file.stem not in EXISTING_EVENTS:
                with open(file, 'rb') as f:
                    db[file.stem] = pickle.load(f)

    time_end = datetime.datetime.now()
    time_delta = int((time_end - time_start).total_seconds())
    print(
        '{}: Directory {} done. Took {} seconds. Using {} GB mem'.format(
            get_time(),
            directory.stem,
            time_delta,
            round(process.memory_info().rss * 1e-9, 2)
        )
    )
