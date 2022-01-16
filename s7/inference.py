from torchvision import models, datasets
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

import time

from ptflops import get_model_complexity_info

resnet = models.resnet152(pretrained=True)
mobilenet = models.mobilenet_v3_large(pretrained=True)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = datasets.CIFAR10("/tmp", train=False, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, num_workers=2)

resnet.eval()
mobilenet.eval()

macs, params = get_model_complexity_info(resnet, tuple(dataset[0][0].shape),
                                         as_strings=False,
                                         print_per_layer_stat=False,
                                         verbose=False)

print('{:<30}  {:<8}'.format('Computational complexity ResNet: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters ResNet: ', params))

macs, params = get_model_complexity_info(mobilenet, tuple(dataset[0][0].shape),
                                         as_strings=False,
                                         print_per_layer_stat=False,
                                         verbose=False)

print('{:<30}  {:<8}'.format('Computational complexity MobilNet V3 Large: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters MobilNet V3 Large: ', params))




start = time.time()
for imgs, labels in dataloader:
    resnet(imgs)
end = time.time()
print(f'Timing ResNet512: {end - start:.2f} s')


start = time.time()
for imgs, labels in dataloader:
    mobilenet(imgs)
end = time.time()
print(f'Timing MobileNet V3: {end - start:.2f} s')
