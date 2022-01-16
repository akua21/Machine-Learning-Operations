"""
LFW dataloading
"""
import argparse
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import os
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image


class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform):
        self.transform = transform

        data = []
        names = os.listdir(path_to_folder)

        for name in names:
            path_to_name = os.path.join(path_to_folder, name)
            images_name = os.listdir(path_to_name)

            for image_name in images_name:
                path_to_image = os.path.join(path_to_name, image_name)

                data.append([name, path_to_image])
        
        self.dataframe = pd.DataFrame(data, columns=["name", "path_image"])
        
    def __len__(self):
        return len(self.dataframe) 
    
    def __getitem__(self, index: int):
        img = Image.open(self.dataframe.loc[index]["path_image"])
        return self.transform(img)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='data/lfw', type=str)
    parser.add_argument('-num_workers', default=0, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    # Note we need a high batch size to see an effect of using many
    # number of workers (512)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False,
                            num_workers=args.num_workers)
    
    if args.visualize_batch:
        imgs = next(iter(dataloader))

        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.show()

        
    if args.get_timing:
        # lets do so repetitions
        res = [ ]
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > 100:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)
        print(f'Timing: {np.mean(res)}+-{np.std(res)}')
