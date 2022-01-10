from torch import nn, optim

from pytorch_lightning import LightningModule

class MyAwesomeModel(LightningModule):
    def __init__(self, hidden_layer, activation, last_activation, output_shape, lr):
        super().__init__()
        self.hidden = nn.Linear(hidden_layer[0], hidden_layer[1])
        self.output = nn.Linear(hidden_layer[1], output_shape)
        self.sigmoid = eval("nn." + activation)
        self.softmax = eval("nn." + last_activation)
        self.flatten = nn.Flatten()

        self.lr = lr
        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x):
        if x.shape[1] != 28 or x.shape[2] != 28:
            raise ValueError('Expected each sample to have shape [batch_size, 28, 28]')
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

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        target = target.long()
        loss = self.criterium(preds, target)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)