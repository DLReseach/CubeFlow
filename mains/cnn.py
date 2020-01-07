import os
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
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=LOG_PATH,
        histogram_freq=1
    )
    if config.wandb == True:
        wandb.init(
                    project='cubeflow',
                    name=experiment_name,
                    sync_tensorboard=True
                )
        callbacks = [
            tensorboard,
            WandbCallback(log_weights=True)   
        ]
    else:
        callbacks = [tensorboard]


    data = CnnSplit(config)
    train, validation, test = data.return_indices()
    train_generator = CnnGenerator(config, train, test=False)
    validation_generator = CnnGenerator(config, validation, test=False)
    test_generator = CnnGenerator(config, test, test=True)
    
    loss = tf.keras.losses.CosineSimilarity(axis=1)
    model = cnn_model(config)
    model.compile(
        optimizer='adam',
        loss='MAE',
        # metrics=['MeanAbsoluteError']
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
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        shuffle=True,
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
        # for i in range(resolution.shape[1]):
        #     print(i)
        #     fig, ax = plt.subplots()
        #     ax.hist(resolution[:, i], bins='auto')
        #     ax.set(
        #         title='Resolution axis {}'.format(i),
        #         xlabel='Resolution',
        #         ylabel='Frequency'
        #     )
        #     wandb.log({'chart': fig})
        fig, ax = plt.subplots()
        ax.hist(direction, bins='auto')
        ax.set(title='Direction', xlabel='Angle (radians)', ylabel='Frequency')
        wandb.log({'chart': fig})

if __name__ == '__main__':
    main()
