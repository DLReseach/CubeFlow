import os
import psutil
import joblib
from cnn.cnn_split import CnnSplit
from cnn.cnn_dataloader import CnnGenerator

process = psutil.Process(os.getpid())

data_type = 'oscnext-genie-level5-v01-01-pass2'
mask = 'SplitInIcePulses'
transform = 'transform1'
features = [
    'dom_charge',
    'dom_time',
    'dom_x',
    'dom_y',
    'dom_z'
]
targets = ['true_primary_time']
test_fraction = 0.2
validation_fraction = 0.2
random_state = 29897070
no_of_files = 1
max_doms = 60
batch_size = 1

test = CnnSplit(
    data_type,
    mask,
    transform,
    test_fraction,
    validation_fraction,
    random_state,
    max_doms,
    no_of_files
)

train, validate, test = test.return_indices()
training_generator = CnnGenerator(
    train,
    features,
    targets,
    max_doms,
    mask,
    transform,
    batch_size,
    len(features),
    shuffle=True
)
blah = next(iter(training_generator))
print(blah)

print(
    'Mem usage: {} GB'.format(round(process.memory_info().rss / 1073741824, 2))
)