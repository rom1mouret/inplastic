This is my first attempt at implementing a neural network in a [lifelong learning](http://lifelongml.org) setting, while avoiding the pitfalls of catastrophic forgetting and [catastrophic interference](https://en.wikipedia.org/wiki/Catastrophic_interference).

As a first step, I want to test the limits of a plain [few-shot learning](https://en.wikipedia.org/wiki/One-shot_learning) approach and see how far it goes without any incremental adjustment of the main model, i.e. by simple [transfer](https://en.wikipedia.org/wiki/Transfer_learning) to unseen classes.

## Starting small

The system will learn to distinguish types of fruit from the [Fruit-360 dataset](https://www.kaggle.com/moltean/fruits).
It consists of 120 classes of fruits and vegetables. Each fruit/vegetable is photographed from over 100 angles.

The world model is built by training a CNN in a [siamese](https://en.wikipedia.org/wiki/Siamese_neural_network) fashion. It takes an image as input and generates a 90-dimensional vector as output.
The training algorithm minimizes intra-class L1 distances between said vectors, while maximizing inter-class distances up to a fixed maximum.

```python
batch1 = one_random_image_for_each_class()
batch2 = one_random_image_for_each_class()
dist = pairwise_distance(batch1, batch2)
loss = mean(diagonal(dist)) + max_dist - mean(min(max_dist, dist))
```

To be clear, the intra-class distance we wish to minimize is included in both `diagonal(dist)` and `mean(min(max_dist, dist))` but it has a comparatively lower weight in the latter operand.


## Memory retrieval

Given a fruit class, we hope that all the description vectors are roughly identical across all the class examples, as expected from optimizing the loss function above.
In practice, while this is true for the most part, the bunch can be spoiled by a few bad apples.

A simple but passable strategy is to compare memorized classes with unknown fruits using [Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance).

<p align="center">
  <img src="/images/implementation.png" width="350">
</p>

My experiments show that a better strategy is to train a univariate RBF Kernel Density Estimator for each dimension of the description vectors.
We'll use negative log density as distance, as provided by [scikit's score_samples function](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity.score_samples).


## Fruitful bagging

The CNN is tested in two configurations.

- The default configuration is a plain CNN.

- In the other configuration, the CNN is divided into 28 "neural columns".

Neural columns are trained on their own small subset of fruit classes, just like bagging.
I believe this prevents the network from reserving output dimensions to specific training classes, which wouldn't be useful to new classes.
I also expect the cosine similarity between description vectors to be higher with the default configuration, since nothing is done to stop the big CNN from generating redundant descriptive features.
This seems to be confirmed experimentally with `cosine_similarity.py`.

## Results

<p align="center">
  <img src="/images/results.png">
</p>

## Reproducibility

Full disclosure: Training and evaluating take a while. I haven't run the full cycle enough times to guarantee a confidence interval.

To reproduce similar results, move the [Fruit-360 dataset directory](https://www.kaggle.com/moltean/fruits) to the root of the repository.
Here is what it looks like when I execute `find . -maxdepth 2 -type d -ls` in my working directory:

```bash
.
./fruits-360
./fruits-360/test-multiple_fruits
./fruits-360/Test
./fruits-360/papers
./fruits-360/Training
./images
```

I have uploaded a pre-trained network. The model can be evaluated right away.

```bash
./test_model.py model_28_14_3
```

Otherwise, refer to `train_model.py --help` to train a new model.


#### Requirements

Tested with Python 3.6.9 on Mint 19.1 but should work with Python >= 3.5 on any OS.

You will find PyPi requirements in `requirements.txt`.

## General approach

Adult human brains are known to be plastic to some degree, so I don't expect basic transfer learning to pave the way for human-level learning abilities.
That being said, there is evidence that some parts of the brain are not plastic enough to [reorganize after brain damage](https://www.youtube.com/watch?v=HPViT0sbJ8o). Hence, there is hope that some core functions of the brain, such as task switching, do not require deep rewiring at adulthood, even when presented with new tasks.

What does it take to incrementally "learn" without deep rewiring?

First, we have to part ways with the traditional deep learning setup wherein new knowledge and experience are stored in the neural connections that translate sensory inputs into actions. In such a setup, acquiring new knowledge interferes with previously learned knowledge.
Instead, our system stores experience in a boring collection of immutable objects, such as an associative array mapping *processed* sensory inputs to actions. Needless to say, associative arrays are interference-free.

Second, a hard-wired world model has to produce descriptions that are extremely rich and fine-grained, manifesting invariance that facilitates generalization and reasoning.
Such descriptions are to be cast in terms of object permanence, hierarchical object structures, 3D physics and so on. 
It is challenging to say the least, but it is arguably *not* a lifelong business.

<p align="center">
  <img src="/images/system.png" width="350">
</p>

I haven't included the compressor in my code.
I imagine the array quickly gets too large for real-life purposes, so you might want to routinely compress its content thanks to yet another hard-wired algorithm.
