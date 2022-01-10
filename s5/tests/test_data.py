import sys
sys.path.insert(1, "src/data/")
from make_dataset import mnist 

input_filepath = "data/raw"
output_filepath = "data/processed"
training, test = mnist(input_filepath, output_filepath)

dataset = training.dataset + test.dataset

def test_size():
    assert len(dataset) == len(training) * training.batch_size + len(test) * test.batch_size, "Dataset did not have the correct number of samples"

def test_shape():
    for img, _ in training:
        assert img.shape == (8, 28, 28), "Image from training set did not have the correct shape"

    for img, _ in test:
        assert img.shape == (8, 28, 28), "Image from test set did not have the correct shape"

def test_labels():
    for _, labels in training:
        for lab in labels:
            assert lab in range(10), "Labels from training set did not have the allowed value"

    for _, labels in test:
        for lab in labels:
            assert lab in range(10), "Labels from test set did not have the allowed value"