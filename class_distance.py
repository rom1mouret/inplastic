from torch import Tensor
import torch
import numpy as np
from sklearn.neighbors import KernelDensity


class BasicDistance:
    def memorize(self, class_filters: Tensor) -> Tensor:
        return class_filters.median(dim=0)[0]

    def distance(self, memorized: Tensor, batch: Tensor) -> Tensor:
        median = memorized.unsqueeze(0)
        return (median - batch).pow(2).sum(dim=1)

    def __repr__(self) -> str:
        return "basic-dist"


class MahalanobisDistance:
    def memorize(self, class_filters: Tensor) -> tuple:
        return class_filters.median(dim=0)[0], class_filters.var(dim=0)

    def distance(self, memorized: tuple, batch: Tensor) -> Tensor:
        median, variance = memorized
        median = median.unsqueeze(0)
        variance = variance.unsqueeze(0)

        return ((median - batch).pow(2) / variance).sum(dim=1)

    def __repr__(self) -> str:
        return "mahalanobis"


class DensityDistance:
    def memorize(self, class_filters: Tensor) -> list:
        array = class_filters.numpy()
        return [
            KernelDensity().fit(array[:, [i]])
            for i in range(class_filters.size(1))
        ]

    def distance(self, memorized: list, batch: Tensor) -> Tensor:
        array = batch.numpy()
        logproba = np.zeros(batch.size(0))
        for i, kde in enumerate(memorized):
            logproba += kde.score_samples(array[:, [i]])

        return -Tensor(logproba)

    def __repr__(self) -> str:
        return "kde-dist"
