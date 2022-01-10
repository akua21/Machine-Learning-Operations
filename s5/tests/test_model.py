import sys
import pytest

import torch
sys.path.insert(1, "src/models/")
from model import MyAwesomeModel

hidden_layer = [784, 256]
activation = "Sigmoid()"
last_activation = "Softmax(dim=1)"
output_shape = 10
lr = 0.0001

model = MyAwesomeModel(hidden_layer, activation, last_activation, output_shape, lr)

def test_output_shape():
    x = torch.rand((8, 28, 28))
    y = model(x)

    assert y.shape == (8, 10), "Output did not have the correct shape"

def test_bad_input_shape():
    with pytest.raises(ValueError, match='Expected each sample to have shape .*'):
        x = torch.rand((8, 30, 30))
        model(x)

@pytest.mark.parametrize("shape", [(1, 28, 28), (2, 28, 28), (5, 28, 28)])
def test_output_shape_multiple(shape):
    x = torch.rand(shape)
    y = model(x)

    assert y.shape == (shape[0], 10), "Output did not have the correct shape"
