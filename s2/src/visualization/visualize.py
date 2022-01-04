import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm

from src.models import MyAwesomeModel

seaborn.set()

model = MyAwesomeModel()
model.load_state_dict(torch.load("models/mnist/trained_model.pt"))
model.eval()


train_set = torch.load("data/processed/corruptmnist/train.pt")

features = []
colors = []

for images, labels in tqdm(train_set):
    out = model.last_layer_features(images)

    features.extend(out.detach().numpy())
    colors.extend(labels.detach().numpy() * 28 / 255)

tsne = TSNE(n_components=2).fit_transform(features)

tx = tsne[:, 0]
ty = tsne[:, 1]

tx = (tx - tx.min()) / (tx.max() - tx.min())
ty = (ty - ty.min()) / (ty.max() - ty.min())


plt.figure(figsize=(8, 8))
plt.title("Train features visualization")


plt.scatter(tx, ty, c=colors)


plt.savefig("reports/figures/visualization.png")
plt.show()
