import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

import wandb as wandb
import numpy as np
import logging
import time
import datetime
from coolname import generate_slug
import matplotlib.pyplot as plt
import matplotlib.cbook
import warnings

from plots.plot_functions import histogram
from data_loader.cnn_generator import CnnGenerator
from models.cnn_model import CnnNet
from preprocessing.cnn_preprocessing import CnnSplit
from utils.config import process_config
from utils.utils import get_args
from utils.utils import get_project_root

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings(
    'ignore',
    category=matplotlib.cbook.mplDeprecation
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    if config.wandb == True:
        wandb.init(
                project='cubeflow',
                name=experiment_name,
                sync_tensorboard=True
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
    train_generator = torch.utils.data.DataLoader(
        CnnGenerator(config, train, test=False),
        batch_size=None
    )
    validation_generator = torch.utils.data.DataLoader(
        CnnGenerator(config, validation, test=False),
        batch_size=None
    )
    test_generator = torch.utils.data.DataLoader(
        CnnGenerator(config, test, test=True),
        batch_size=None
    )
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

    model = CnnNet(config)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(config.num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_generator, 0):
            if i == 0:
                ts = time.time()
                st = datetime.datetime.fromtimestamp(ts).strftime(
                    '%Y-%m-%d %H:%M:%S'
                )
                print(
                    'Starting epoch {}/{} at {}'
                    .format(
                        epoch,
                        config.num_epochs,
                        st
                    )
                )
            inputs, targets = data
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 1))
                running_loss = 0.0

    if config.wandb == True:
        wandb.init(
                    project='cubeflow',
                    name=experiment_name
                )


    # def unit_vector(vector):
    #     """ Returns the unit vector of the vector.  """
    #     return vector / np.linalg.norm(vector)

    # def angle_between(v1, v2):
    #     """ Returns the angle in radians between vectors 'v1' and 'v2'::

    #             >>> angle_between((1, 0, 0), (0, 1, 0))
    #             1.5707963267948966
    #             >>> angle_between((1, 0, 0), (1, 0, 0))
    #             0.0
    #             >>> angle_between((1, 0, 0), (-1, 0, 0))
    #             3.141592653589793
    #     """
    #     v1_u = unit_vector(v1)
    #     v2_u = unit_vector(v2)
    #     return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


    # resolution = np.empty((0, len(config.targets)))
    # direction = np.empty((0, 1))
    # for X, y_truth in test_generator:
    #     y_predict = model.predict_on_batch(X)
    #     resolution = np.vstack([resolution, (y_truth - y_predict.numpy())])
    #     for i in range(y_predict.shape[0]):
    #         angle = angle_between(y_truth[i, :], y_predict.numpy()[i, :])
    #         direction = np.vstack([direction, angle])

    # if config.wandb == True:
    #     fig, ax = histogram(
    #         data=direction,
    #         title='y_truth . y_pred / (||y_truth|| ||y_pred||)',
    #         xlabel='Angle (radians)',
    #         ylabel='Frequency',
    #         width_scale=1,
    #         bins='fd'    
    #     )
    #     wandb.log({'chart': fig})

if __name__ == '__main__':
    main()
