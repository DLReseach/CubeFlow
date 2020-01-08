import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
import wandb as wandb
from wandb.keras import WandbCallback
import numpy as np
import logging
import time
import datetime
from coolname import generate_slug
import matplotlib.pyplot as plt
import matplotlib.cbook
import warnings

from plots.plot_functions import histogram
from callbacks.cnn_callbacks import cnn_callbacks
from data_loader.cnn_generator import CnnGenerator
from models.cnn_model import cnn_model
from preprocessing.cnn_preprocessing import CnnSplit
from utils.config import process_config
from utils.utils import get_args
from utils.utils import get_project_root

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings(
    'ignore',
    category=matplotlib.cbook.mplDeprecation
)


class EpochDurationCallback(tf.keras.callbacks.Callback):
    def __init__(self, no_of_batches, validation_generator):
        self.times = []
        self.times.append(time.time())
        self.no_of_batches = no_of_batches
        self.validation_generator = validation_generator
    

    def on_epoch_begin(self, epoch, logs=None):
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        print('\nStarting epoch {0} at {1}'.format(epoch + 1, st))


    def on_epoch_end(self, epoch, logs=None):
        timestamp = time.time()
        duration = timestamp - self.times[-1]
        self.times.append(timestamp)
        print('\nEpoch lasted {0:.2f} minutes'.format(duration / 60))
    

    def on_train_batch_end(self, batch, logs=None):
        if ((batch + 1) % 1000 == 0) or (batch == 0):
            val_loss = self.model.evaluate_generator(
                generator=self.validation_generator,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False,
                verbose=0
            )
            wandb.log(
                {
                    'loss': logs['loss'],
                    'cosine_similarity': logs['cosine_similarity'],
                    'val_loss': val_loss
                }
            )



def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print('missing or invalid arguments')
        exit(0)

    PARAMS = {
        'lr': config.learning_rate,
        'momentum': config.momentum
    }
    root_folder = get_project_root()
    cool_name = generate_slug(2)
    experiment_name = config.exp_name \
        + '_' + str(datetime.date.today()) + '.' + cool_name
    model_plot_file = root_folder.joinpath(
        './figures/' + experiment_name + '.png'
    )
    LOG_PATH = root_folder.joinpath('logs/fit' + experiment_name)
    LOG_PATH.mkdir(exist_ok=True, parents=True)

    if config.wandb == True:
        wandb.init(
                project='cubeflow',
                name=experiment_name,
                sync_tensorboard=True
            )

    print(
        'Num GPUs Available: ',
        len(tf.config.experimental.list_physical_devices('GPU'))
    )

    ts1 = time.time()
    st1 = datetime.datetime.fromtimestamp(ts1).strftime('%Y-%m-%d %H:%M:%S')
    print('Starting preprocessing at {}'.format(st1))
    data = CnnSplit(config)
    train, validation, test = data.return_indices()
    ts2 = time.time()
    st2 = datetime.datetime.fromtimestamp(ts2).strftime('%Y-%m-%d %H:%M:%S')
    print('Ended preprocessing at {}'.format(st2))
    td = ts2 - ts1
    td_secs = int(td)
    print('Preprocessing took approximately {} seconds'.format(td_secs))
    train_generator = CnnGenerator(config, train, test=False)
    validation_generator = CnnGenerator(config, validation, test=False)
    test_generator = CnnGenerator(config, test, test=True)
    print(
        'We have around {} training events'.format(
            len(train_generator) * config.batch_size
        )
    )
    print(
        'We have around {} validation events'.format(
            len(validation_generator) * config.batch_size
        )
    )
    print(
        'We have around {} test events'.format(
            len(test_generator) * config.batch_size
        )
    )

    np.random.seed(int(time.time()))

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=LOG_PATH,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
        embeddings_data=None,
        update_freq=1000,
        profile_batch=3
    )

    if config.wandb == True:
        wandb.init(
                    project='cubeflow',
                    name=experiment_name,
                    sync_tensorboard=True
                )
        callbacks = [
            EpochDurationCallback(
                no_of_batches=len(train_generator) + 1,
                validation_generator=validation_generator
            ),
            tensorboard,
            WandbCallback(log_weights=True)   
        ]
    else:
        callbacks = [
            EpochDurationCallback(
                no_of_batches=len(train_generator) + 1,
                validation_generator=validation_generator
            ),
            tensorboard
        ]

    cosine_metric = tf.keras.losses.CosineSimilarity(axis=1)
    model = cnn_model(config)
    model.compile(
        optimizer='adam',
        loss='MSE',
        metrics=[cosine_metric]
    )

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=np.ceil(len(train) / config.batch_size),
        epochs=config.num_epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=None,
        validation_freq=1,
        class_weight=None,
        max_queue_size=100,
        workers=4,
        use_multiprocessing=True,
        shuffle=False,
        initial_epoch=0
    )


    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


    resolution = np.empty((0, len(config.targets)))
    direction = np.empty((0, 1))
    for X, y_truth in test_generator:
        y_predict = model.predict_on_batch(X)
        resolution = np.vstack([resolution, (y_truth - y_predict.numpy())])
        for i in range(y_predict.shape[0]):
            angle = angle_between(y_truth[i, :], y_predict.numpy()[i, :])
            direction = np.vstack([direction, angle])

    if config.wandb == True:
        fig, ax = histogram(
            data=direction,
            title='y_truth . y_pred / (||y_truth|| ||y_pred||)',
            xlabel='Angle (radians)',
            ylabel='Frequency',
            width_scale=1,
            bins='fd'    
        )
        wandb.log({'chart': fig})

if __name__ == '__main__':
    main()
