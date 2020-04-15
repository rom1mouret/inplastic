#!/usr/bin/env python3

import sys
import os
import torch
import numpy as np

from neural_net import Ensemble
from data import train_selector, test_selector1, test_selector2, load_images
from class_distance import BasicDistance, MahalanobisDistance, DensityDistance

np.random.seed(0)
torch.manual_seed(0)

if len(sys.argv) <= 1 or sys.argv[1] in ("-h", "--help"):
    print("Usage: %s model-path [basic | density | mahalanobis]" % sys.argv[0])
    exit(1)

# load PyTorch model
path = sys.argv[1]
parts = path.split("_")

columns, channels, col_dim = parts[-3:]
columns = int(columns)
channels = int(channels)
col_dim = int(col_dim)

net = Ensemble(n_columns=columns, column_dim=col_dim, channels=channels)
net.load_state_dict(torch.load(path))
net.eval()
for param in net.parameters():
    param.requires_grad_(False)

# choose between distances
distance = MahalanobisDistance()
if len(sys.argv) >= 3:
    dist = sys.argv[2].lower()
    if dist == "basic":
        distance = BasicDistance()
    elif dist == "density":
        distance = DensityDistance()
else:
    dist = "mahalanobis"

print("distance choice:", distance)

# load examples
def read_classes():
    for selector in (train_selector, test_selector1, test_selector2):
        for klass in load_images(selector, max_per_class=64):
            yield klass

        print("--- end of selector ---")


# run model as we populate the memory
examples_per_class = 16
eval_examples = {}
memory = {}
accuracy = []
for class_i, examples in enumerate(read_classes()):
    print("evaluation of class", class_i, "(", len(examples), "examples )")
    examples = torch.cat(examples, dim=0)

    # memorize the class statistics
    filters = net(examples)
    memory[class_i] = distance.memorize(filters)

    # keep some examples for evaluation
    eval_examples[class_i] = filters[:examples_per_class, :]

    if class_i == 0:
        continue  # nothing to evaluate yet

    # evaluation
    total = 0
    hits = 0
    for class_j, class_j_img in eval_examples.items():
        dist = np.empty((len(eval_examples), len(class_j_img)))
        for class_k in eval_examples.keys():
            dist[class_k, :] = \
                distance.distance(memory[class_k], class_j_img).numpy()

        closest = dist.argmin(axis=0)
        total += len(closest)
        hits += (closest == class_j).sum()
    print("accuracy", hits/total)
    accuracy.append(hits/total)

with open("accuracy_%s_%s.txt" % (os.path.basename(path), distance), "w") as f:
    for acc in accuracy:
        f.write("%f\n" % acc)

for i, acc in enumerate(accuracy):
    print("#%i accuracy=%.2f" % (i, 100*acc))
