import shutil
from pathlib import Path

SOURCE_ROOT = Path.home().joinpath('small_data_test/oscnext-genie-level5-v01-01-pass2/pickles')

SOURCE_DIRS = sorted([
    folder for folder in SOURCE_ROOT.iterdir() if folder.is_dir()
], reverse=True)[0:50]

for source_dir in SOURCE_DIRS:
    print('Deleting', source_dir)
    shutil.rmtree(source_dir)

print('Done!')
