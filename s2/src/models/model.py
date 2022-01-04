from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(784, 256)
        self.output = nn.Linear(256, 10)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)

        return x

    def last_layer_features(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.sigmoid(x)

        return x
