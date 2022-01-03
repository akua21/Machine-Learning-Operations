# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms


def mnist(input_filepath, output_filepath, batch_size=8):
    files_folder = f"{input_filepath}/corruptmnist/"

    # Train
    train_files = [f"{files_folder}train_{i}.npz" for i in range(5)] 

    x_train = []
    y_train = []

    for file in train_files:
        with np.load(file) as data:
            x_train.extend(data["images"])
            y_train.extend(data["labels"])

    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)

    # Test
    test_file = "test.npz"
    x_test = []
    y_test = []

    with np.load(files_folder + test_file) as data:
        x_test = data["images"]
        y_test = data["labels"]

    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)


    # Define a transform to normalize the data
    transform = transforms.Normalize((0.5,), (0.5,))

    trainset = TensorDataset(transform(x_train), y_train)
    train = DataLoader(trainset, shuffle=True, batch_size=batch_size)

    testset = TensorDataset(transform(x_test), y_test)
    test = DataLoader(testset, shuffle=False, batch_size=batch_size)

    torch.save(train, f"{output_filepath}/corruptmnist/train.pt")
    torch.save(test, f"{output_filepath}/corruptmnist/test.pt")

    return train, test





@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    mnist(input_filepath, output_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    
    main()