import tensorflow as tf
from utils.utils import get_project_root

summary_file = get_project_root().joinpath(
    'logs/fit/cnn_2019-12-29.orthodox-flounder/train/'
    'events.out.tfevents.1577631234.work.12677.204.v2'
)
blah = tf.compat.v1.train.summary_iterator(str(summary_file))

for e in blah:
    print(type(e))
