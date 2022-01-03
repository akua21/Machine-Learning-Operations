import argparse
import sys

import torch

from data import mnist
from model import MyAwesomeModel
from torch import nn
from torch import optim

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.0001)
        parser.add_argument('--epochs', default=10)
        parser.add_argument('--save_model_to', default='trained_model.pt')
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        model.train()
        train_set, _ = mnist()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        plot_loss = []

        for e in range(args.epochs):
            running_loss = 0
            for images, labels in train_set:
                optimizer.zero_grad()

                outputs = model(images)
                labels = labels.long()

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

            print(f"Epoch: {e+1}/{args.epochs}\t Loss: {running_loss/len(train_set):5f}")
            plot_loss.append(running_loss/len(train_set))
        

        torch.save(model.state_dict(), args.save_model_to)

        plt.figure(figsize=(12, 8))
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.xticks(range(1, args.epochs+2))
        plt.ylabel("Loss")
        plt.plot(range(1, args.epochs+1), plot_loss)
        plt.savefig("training_loss.png")
        plt.show()



        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        # model = torch.load(args.load_model_from)
        model = MyAwesomeModel()
        model.load_state_dict(torch.load(args.load_model_from))
        model.eval()
        _, test_set = mnist()

        running_acc = 0

        for images, labels in test_set:
            outputs = model(images)
            labels = labels.data.numpy()

            out_data = torch.max(outputs, 1)[1].data.numpy()

            running_acc += accuracy_score(labels, out_data)
        
        print(f"Model accuracy: {running_acc/len(test_set)*100:.2f}%")

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    