import torch

import numpy as np
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader


def mnist(batch_size=8):
    # exchange with the corrupted mnist dataset
    # train = torch.randn(50000, 784)
    # test = torch.randn(10000, 784) 


    files_folder = "./corruptmnist/"

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

    return train, test
