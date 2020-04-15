#!/usr/bin/env python3

import argparse
import random
import numpy as np
from tqdm import tqdm
import torch

from data import train_selector, test_selector1, load_images
from neural_net import Ensemble

np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser(description="Siamese network training on CPU")
parser.add_argument('--columns', metavar="n", type=int, nargs=None,
        help="number of neural columns, default: 28", default=28)
parser.add_argument('--channels', metavar="n", type=int, nargs=None,
    help="number of hidden CNN channels in columns, default: 14", default=14)
parser.add_argument('--classes-per-col', metavar="n", type=int, nargs=None,
    help="number of classes per columnm default: 16", default=16)
args = vars(parser.parse_args())


# load validation data
class_images = load_images(test_selector1, max_per_class=32)
validation = [
    torch.cat(class_images[i], dim=0)
    for i in range(4)
]

# load main training set
class_images = load_images(train_selector, max_per_class=256)
print("number of classes:", len(class_images))

# a function to generate processed batch
def random_batch(net: torch.nn.Module, class_indices: list) -> torch.Tensor:
    inp = torch.cat([
        random.choice(class_images[j]) for j in class_indices
    ], dim=0)
    return net(inp)

# ensemble of neural columns
classes_per_model = args["classes_per_col"]
dim = int(np.ceil(np.log(classes_per_model)))
ensemble = Ensemble(
    n_columns=args["columns"], column_dim=dim, channels=args["channels"])
nets = ensemble.columns
optimizers = [
    torch.optim.Adam(net.parameters(), lr=0.01)
    for net in nets
]

# choose a random set of classes for each network
class_sets = [
    np.random.permutation(len(class_images))[:classes_per_model]
    for _ in nets
]

# this distance function will be handy for the loss function
def distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    diff = x.unsqueeze(0) - y.unsqueeze(1)
    return diff.abs().mean(dim=2)

# where we will store the best validated ensemble
filename = "ensemble_{columns}_{channels}_{col_dim}".format(
    columns=ensemble.num_columns(),
    channels=ensemble.channels(),
    col_dim=ensemble.column_dim()
)
print("model's weights will be stored in", filename)

# training parameters
max_dist = 5.0
best_loss = np.inf
iterations_per_epoch = 10

# training loop
print("end training with CTRL-C whenever you see fit")
for epoch in range(1000):
    progress_bar = tqdm(range(iterations_per_epoch), total=iterations_per_epoch)
    for _ in progress_bar:
        for net, class_set, optimizer in zip(nets, class_sets, optimizers):
            # batch of one random image for each class
            batch1 = random_batch(net, class_set)
            batch2 = random_batch(net, class_set)

            # distance loss
            dist = distance(batch1, batch2)
            class_indices = list(range(len(class_set)))
            intra_dist = dist[class_indices, class_indices].mean()
            inter_dist = dist.clamp(max=max_dist).mean()
            loss = intra_dist + max_dist - inter_dist

            # optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # validation
    all_classes = list(range(len(class_images)))
    with torch.no_grad():
        batch1 = random_batch(ensemble, all_classes)
        batch2 = random_batch(ensemble, all_classes)
        dist = distance(batch1, batch2)
        intra_dist = dist[all_classes, all_classes].mean()
        inter_dist = dist.clamp(max=max_dist).mean()
        loss = (intra_dist + max_dist - inter_dist).item()
        if loss < best_loss:
            best_loss = loss
            torch.save(ensemble.state_dict(), filename)
            print("[epoch %i] best loss = %f" % (epoch, best_loss))
