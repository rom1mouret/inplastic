#!/usr/bin/env python3

import sys
import torch
import numpy as np

from neural_net import load_model
from data import train_selector, test_selector1, test_selector2, load_images

np.random.seed(0)
torch.manual_seed(0)

net = load_model(sys.argv[1])
images = []
for selector in (train_selector, test_selector1, test_selector2):
    images += load_images(selector,  max_per_class=1)[0]

batch = net(torch.cat(images, dim=0))

transpose = batch.t()
norms = transpose.pow(2).sum(dim=1).clamp(min=0.001).sqrt()
norm_mat = norms.unsqueeze(0) * norms.unsqueeze(1)
redundancy = ((transpose @ batch) / norm_mat).abs().mean()

print("redudancy", redundancy)
