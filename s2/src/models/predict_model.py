import argparse
import os
import sys

import numpy as np
import torch
from model import MyAwesomeModel
from PIL import Image
from torchvision import transforms


class Evaluate(object):
    """Helper class that will help launch class methods as commands
    from a single script
    """

    def __init__(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("load_model_from", default="")
        parser.add_argument("load_data_from", default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[1:])
        print(args)

        model = MyAwesomeModel()
        test_set = None
        transform = transforms.ToTensor()

        # Images
        if os.path.isdir(args.load_data_from):
            test_set = []
            for img in os.listdir(args.load_data_from):
                image = Image.open(os.path.join(args.load_data_from, img))
                image = image.convert("L")
                test_set.append(transform(image))
        # npy files
        else:
            loaded_data = np.load(args.load_data_from)
            test_set = [transform(x.reshape((28, 28, 1))) for x in loaded_data]

        model.load_state_dict(torch.load(args.load_model_from))
        model.eval()

        index = 0
        for images in test_set:
            outputs = model(images)

            out_data = torch.max(outputs, 1)[1].data.numpy()

            for out in out_data:
                print(f"Prediction of input data {index} : {out}")
                index += 1


if __name__ == "__main__":
    Evaluate()
