# Deep Learning specialization by Deeplearning.ai

## Neural Networks & Deep Learning
The coursework provides a foundational understanding of a neural network. Different NN-based classifier models were built from scratch, using first principles - forward & backward propagation, calculating gradients and optimizing using gradient descent approach, **using mostly NumPy**. Classifiers built:
1. Logistic regression classifier (as a single layer neural network) - test accuracy of ~70% [notebook](https://github.com/uditgt/DeepLearning/blob/main/Logistic%20Regression%20from%20scratch.ipynb)
2. Shallow neural network - test accuracy of ~72% [notebook](https://github.com/uditgt/DeepLearning/blob/main/Neural%20Network%20from%20scratch.ipynb)
3. Deep neural network - test accuracy of 82% [notebook](https://github.com/uditgt/DeepLearning/blob/main/Deep%20Neural%20Network%20from%20scratch.ipynb)

Focus of these exercises was on carefully piecing together the different equations for forward and backward pass, in vectorized form, to build a functioning neural network.

## Improving Deep Neural Networks
The course focuses on how to tune a neural networks, and discusses hyper-parameter tuning, regularization and optimization methods in more details. Including various approaches for **Regularization**, such as L2, drop-out, data augmentation; for **Optimization**, such as normalizing inputs, batch normalization, initializing neurons using Xavier method to manage variance to avoid vanishing/ exploding gradients; improving **Gradient Descent** algorithm by using mini-batches, Momentum / RMSprop / Adam methods, learning rate decay; more effectively tuning **Hyper-parameters** using random sampling approach (vs. grid search), prioritizing certain parameters over others, using appropriate scale for each hyper-parameter.

## Structuring Machine Learning Projects
The courses focuses on lots of practical do's and dont's when working on a ML project, including metric selection, structuring different datasets (train, train-dev, dev, test), analysing errors, addressing data mismatch, transfer learning, multi-task learning, structuring problems as end-to-end deep learning model vs. breaking into smaller parts. 

## Convolution Neural Networks
This course takes on whirlwind tour into the fastastic world of CNNs. Provides a foundational understanding manipulating 3D /volume data, how convolution and pooling operations work, what different layers mean and represent. After coding Forward and Backward propogation functions from scratch (using just numpy), we learn TensorFlow's - [Sequential API](https://www.tensorflow.org/guide/keras/sequential_model) and [Functional API](https://www.tensorflow.org/guide/keras/functional). 

Next we go over various architectures published over the years (and use some through transfer learning) - LeNet, AlexNet, VGG, ResNet, Inception Networks, MobileNet - learning various tricks developed in these networks. Object localization and detection tasks, including learning about bounding box, sliding window through convolution, IoU, non-max suppression, semantic segmentation using YOLO, U-Net architectures. Next we look at face recognition tasks using Siamese Network, DeepFace, FaceNet. Finally, we look at Neural Style Transfer architecture used for merging 'content' and 'style' images. 


## References:
* Dropout regularization, Gofrrey Hinton (2014) [link](https://jmlr.org/papers/v15/srivastava14a.html)
* Xavier initialization (2010) [link](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
