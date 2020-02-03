import shutil
from pathlib import Path

SOURCE_ROOT = Path.home().joinpath('data/oscnext-genie-level5-v01-01-pass2/pickles')
TARGET_ROOT = Path.home().joinpath('small_data_test/oscnext-genie-level5-v01-01-pass2/pickles')

SOURCE_DIRS = sorted([
    folder for folder in SOURCE_ROOT.iterdir() if folder.is_dir()
])[0:400]
TARGET_DIRS = sorted([
    folder.stem for folder in TARGET_ROOT.iterdir() if folder.is_dir()
])

for source_dir in SOURCE_DIRS:
    if source_dir.stem not in TARGET_DIRS:
        print('Copying', source_dir)
        target_dir = TARGET_ROOT.joinpath(source_dir.stem)
        # target_dir.mkdir(exist_ok=False)
        shutil.copytree(source_dir, target_dir)

print('Done!')
