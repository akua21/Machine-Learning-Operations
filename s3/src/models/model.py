from torch import nn

class MyAwesomeModel(nn.Module):
    def __init__(self, hidden_layer, activation, last_activation, output_shape):
        super().__init__()

        self.hidden = nn.Linear(hidden_layer[0], hidden_layer[1])
        self.output = nn.Linear(hidden_layer[1], output_shape)
        self.sigmoid = eval("nn." + activation)
        self.softmax = eval("nn." + last_activation)

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
