import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from torchsummary import summary
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

warnings.filterwarnings(
    'ignore',
    category=matplotlib.cbook.mplDeprecation
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device', device)


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
                name=experiment_name
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
        batch_size=None,
        num_workers=config.num_workers
    )
    validation_generator = torch.utils.data.DataLoader(
        CnnGenerator(config, validation, test=False),
        batch_size=None,
        num_workers=config.num_workers
    )
    test_generator = torch.utils.data.DataLoader(
        CnnGenerator(config, test, test=True),
        batch_size=None,
        num_workers=config.num_workers
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
    model.to(device)
    summary(model, input_size=(len(config.features), config.max_doms))
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    if config.wandb == True:
        wandb.watch(model)

    print_interval = int(np.ceil(len(train_generator) * 0.1))


    def train_step(model, inputs, targets, loss_fn, optimizer):
        model.train()
        predictions = model(inputs)
        loss = loss_fn(targets, predictions)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()


    def validation_step(model, inputs, targets, loss_fn, metric):
        with torch.no_grad():
            model.eval()
            predictions = model(inputs)
            loss = loss_fn(targets, predictions)
            metric_val = torch.mean(metric(targets, predictions))
            return loss.item(), metric_val


    def prediction_step(model, inputs):
        with torch.no_grad():
            model.eval()
            predictions = model(inputs)
            return predictions


    for epoch in range(config.num_epochs):
        running_loss = 0.0
        val_loss = 0.0
        cosine_similarity = 0.0
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime(
            '%Y-%m-%d %H:%M:%S'
        )
        print(
            'Starting epoch {}/{} at {}'
            .format(
                epoch + 1,
                config.num_epochs,
                st
            )
        )
        for i, data in enumerate(train_generator, 0):
            inputs, targets = data
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            loss = train_step(model, inputs, targets, criterion, optimizer)
            running_loss += loss
            if i % print_interval == (print_interval - 1):
                print(
                    '[%d, %5d / %5d] loss: %.3f' %
                    (
                        epoch + 1,
                        i + 1,
                        len(train_generator),
                        running_loss / print_interval
                    )
                )
                for inputs, targets in validation_generator:
                    inputs = inputs.float().to(device)
                    targets = targets.float().to(device)
                    validation_values = validation_step(
                        model,
                        inputs,
                        targets,
                        criterion,
                        torch.nn.CosineSimilarity()
                    )
                    val_loss += validation_values[0]
                    cosine_similarity += validation_values[1]
                if config.wandb == True:
                    wandb.log(
                        {
                            'loss': running_loss / print_interval,
                            'val_loss': val_loss / len(validation_generator),
                            'cosine_similarity': cosine_similarity / len(validation_generator)
                        }
                    )
                running_loss = 0.0
                val_loss = 0.0
                cosine_similarity = 0.0
        for i, data in enumerate(validation_generator, 0):
            inputs, targets = data
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            validation_values = validation_step(
                model,
                inputs,
                targets,
                criterion,
                torch.nn.CosineSimilarity()
            )
            val_loss += validation_values[0]
            cosine_similarity += validation_values[1]
        print(
            'Validation loss epoch {}: {:.3f}'
            .format(
                epoch + 1,
                val_loss / len(validation_generator)
            )
        )
        print(
            'Epoch {} took {:.2f} minutes'
            .format(
                epoch + 1,
                (time.time() - ts) / 60
            )
        )


    def unit_vector(vector):
        ''' Returns the unit vector of the vector.  '''
        return vector / np.linalg.norm(vector)

    def angle_between(v1, v2):
        ''' Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        '''
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


    resolution = np.empty((0, len(config.targets)))
    direction = np.empty((0, 1))
    for inputs, targets in test_generator:
        inputs = inputs.float().to(device)
        targets = targets.float().to(device)
        predictions = prediction_step(model, inputs)
        resolution = np.vstack([resolution, (targets.cpu() - predictions.cpu())])
        for i in range(predictions.shape[0]):
            angle = angle_between(targets.cpu()[i, :], predictions.cpu()[i, :])
            direction = np.vstack([direction, angle])

    if config.wandb == True:
        fig, ax = histogram(
            data=direction,
            title='arccos[y_truth . y_pred / (||y_truth|| ||y_pred||)]',
            xlabel='Angle (radians)',
            ylabel='Frequency',
            width_scale=1,
            bins='fd'
        )
        wandb.log({'chart': fig})

if __name__ == '__main__':
    main()
