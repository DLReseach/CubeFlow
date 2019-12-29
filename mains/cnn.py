import os
from comet_ml import Experiment
import tensorflow as tf
import numpy as np
import logging
from datetime import date
from coolname import generate_slug

from callbacks.cnn_callbacks import cnn_callbacks
from data_loader.cnn_generator import CnnGenerator
from models.cnn_model import cnn_model
from preprocessing.cnn_preprocessing import CnnSplit
from utils.config import process_config
from utils.utils import get_args
from utils.utils import get_project_root

logger = tf.get_logger()
logger.setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
comet_api_key = os.environ['COMET_API_KEY']


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print('missing or invalid arguments')
        exit(0)

    root_folder = get_project_root()
    cool_name = generate_slug(2)
    experiment_name = config.exp_name + '_' + str(date.today()) + '.' + cool_name
    model_plot_file = root_folder.joinpath(
        './figures/' + experiment_name + '.png'
    )

    data = CnnSplit(config)
    train, validation, test = data.return_indices()
    train_generator = CnnGenerator(config, train)
    validation_generator = CnnGenerator(config, validation)
    test_generator = CnnGenerator(config, test)
    
    model = cnn_model(config)
    model.compile(
        optimizer='adam',
        loss='MAE',
        metrics=['MeanAbsoluteError']
    )
    tf.keras.utils.plot_model(
        model,
        to_file=model_plot_file,
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=False,
        dpi=96
    )
    
    experiment = Experiment(
        api_key=comet_api_key,
        project_name='cubeflow',
        workspace='ehrhorn'
    )
    experiment.set_name(experiment_name)
    experiment.log_image(
        image_data=model_plot_file,
        name='Model'
    )

    callbacks = cnn_callbacks(model, config, experiment)

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

    model_plot_file.unlink()


if __name__ == '__main__':
    main()
