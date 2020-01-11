import os
import torch
from torchsummary import summary
import wandb as wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook
import warnings

from data_loader.cnn_generator import CnnGenerator
from models.cnn_model import CnnNet
from preprocessing.cnn_preprocessing import CnnSplit
from utils.config import process_config
from utils.utils import get_args
from utils.utils import get_project_root
from utils.utils import get_time
from utils.utils import print_data_set_sizes
from utils.utils import create_experiment_name
from utils.utils import set_random_seed
from utils.math_funcs import angle_between
from plots.plot_functions import histogram
from plots.plot_functions import matplotlib_histogram
from plots.create_distribution_histograms import DistributionHistograms

warnings.filterwarnings(
    'ignore',
    category=matplotlib.cbook.mplDeprecation
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device', device)


# @profile
def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print('missing or invalid arguments')
        exit(0)

    experiment_name = create_experiment_name(config, slug_length=2)

    if config.wandb == True:
        wandb.init(
                project='cubeflow',
                name=experiment_name
            )

    print('Starting preprocessing at {}'.format(get_time()))
    data = CnnSplit(config)
    (train_df, validation_df, test_df), (train, validation, test) = data.return_indices()
    dist_hists = DistributionHistograms(train_df, validation_df, test_df, config)
    train_hists = dist_hists.create_histograms(train_df)    
    print('Ended preprocessing at {}'.format(get_time()))

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
    print_data_set_sizes(
        config,
        train_generator,
        validation_generator,
        test_generator
    )

    set_random_seed()

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
        print(
            'Starting epoch {}/{} at {}'
            .format(
                epoch + 1,
                config.num_epochs,
                get_time()
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
        fig1, ax1 = matplotlib_histogram(
            data=train_hists['dom_x'],
            title='dom_x train distribution',
            xlabel='dom_x [m]',
            ylabel='Frequency',
            bins='fd'
        )
        wandb.log({'chart': fig1})

if __name__ == '__main__':
    main()
