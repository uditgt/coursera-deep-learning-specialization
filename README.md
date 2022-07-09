# Deep Learning specialization by Deeplearning.ai

## [Neural Networks & Deep Learning](https://www.coursera.org/account/accomplishments/verify/J8L6HGRSHT9C)
The coursework provides a foundational understanding of a neural network. Different NN-based classifier models were built from scratch, using first principles - forward & backward propagation, calculating gradients and optimizing using gradient descent approach, **using mostly NumPy**. Classifiers built:
1. Logistic regression classifier (as a single layer neural network) - test accuracy of ~70% [notebook](https://github.com/uditgt/DeepLearning/blob/main/Logistic%20Regression%20from%20scratch.ipynb)
2. Shallow neural network - test accuracy of ~72% [notebook](https://github.com/uditgt/DeepLearning/blob/main/Neural%20Network%20from%20scratch.ipynb)
3. Deep neural network - test accuracy of 82% [notebook](https://github.com/uditgt/DeepLearning/blob/main/Deep%20Neural%20Network%20from%20scratch.ipynb)

Focus of these exercises was on carefully piecing together the different equations for forward and backward pass, in vectorized form, to build a functioning neural network.

## [Improving Deep Neural Networks](https://www.coursera.org/account/accomplishments/verify/5JGSJWF2WA4Z)
The course focuses on how to tune a neural networks, and discusses hyper-parameter tuning, regularization and optimization methods in more details. Including various approaches for **Regularization**, such as L2, drop-out, data augmentation; for **Optimization**, such as normalizing inputs, batch normalization, initializing neurons using Xavier method to manage variance to avoid vanishing/ exploding gradients; improving **Gradient Descent** algorithm by using mini-batches, Momentum / RMSprop / Adam methods, learning rate decay; more effectively tuning **Hyper-parameters** using random sampling approach (vs. grid search), prioritizing certain parameters over others, using appropriate scale for each hyper-parameter.


## References:
* Dropout regularization, Gofrrey Hinton (2014) [link](https://jmlr.org/papers/v15/srivastava14a.html)
* Xavier initialization (2010) [link](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
