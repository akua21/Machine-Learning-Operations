import matplotlib.pyplot as plt
import seaborn as sns
import torch
from model import MyAwesomeModel
from torch import nn, optim

import hydra
from hydra.utils import get_original_cwd
import random
import numpy as np
import logging

sns.set()
log = logging.getLogger(__name__)


@hydra.main(config_name="config.yaml", config_path=".")
def main(cfg):
    Train(cfg)



class Train(object):
    """Class for training the model.

    It loads a dataset, trains the model and saves it.

    Displays and saves a figure with the loss function from the training.
    """

    def __init__(self, cfg):
        # Training Hyperparameters
        config = cfg.configs
        torch.manual_seed(config.exp.seed)
        random.seed(config.exp.seed)
        np.random.seed(config.exp.seed)

        dataset_path = get_original_cwd() + config.exp.dataset_path
        lr = config.exp.lr
        epochs = config.exp.epochs
        trained_model_path = get_original_cwd() + config.exp.trained_model_path
        training_loss_img_path = get_original_cwd() + config.exp.training_loss_img_path


        
        # Model Hyperparameters
        hidden_layer = config.MyAwesomeModel.hidden_layer
        activation = config.MyAwesomeModel.activation
        last_activation = config.MyAwesomeModel.last_activation
        output_shape = config.MyAwesomeModel.output_shape

        log.info("Training day and night")
        model = MyAwesomeModel(hidden_layer, activation, last_activation, output_shape)
        model.train()
        train_set = torch.load(dataset_path)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        plot_loss = []

        for e in range(epochs):
            running_loss = 0
            for images, labels in train_set:
                optimizer.zero_grad()

                outputs = model(images)
                labels = labels.long()

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            log.info(
                f"Epoch: {e+1}/{epochs}\t Loss: {running_loss/len(train_set):5f}"
            )
            plot_loss.append(running_loss / len(train_set))

        torch.save(model.state_dict(), trained_model_path)
        torch.save(model.state_dict(), trained_model_path.split("/")[-1])

        plt.figure(figsize=(12, 8))
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.xticks(range(1, epochs + 2))
        plt.ylabel("Loss")
        plt.plot(range(1, epochs + 1), plot_loss)
        plt.savefig(training_loss_img_path)
        plt.savefig(training_loss_img_path.split("/")[-1])
        plt.close()


if __name__ == "__main__":
    main()