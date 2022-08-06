# [Deep Learning specialization by Deeplearning.ai](https://www.coursera.org/account/accomplishments/specialization/certificate/Y23QW2JU39ZE)

[Lecture Notes](https://github.com/uditgt/coursera_deeplearning_specialization/tree/main/notes) | [Quizzes](https://github.com/uditgt/coursera_deeplearning_specialization/tree/main/quizzes) | [Paper Review & Summaries](https://github.com/uditgt/coursera_deeplearning_specialization/tree/main/papers) | [Certification](https://www.coursera.org/account/accomplishments/specialization/certificate/Y23QW2JU39ZE)

## Neural Networks & Deep Learning
The coursework provides a foundational understanding of a neural network. Different NN-based classifier models were built from scratch, using first principles - forward & backward propagation, calculating gradients and optimizing using gradient descent approach, **using mostly NumPy**. Classifiers built:
1. Building Logistic regression classifier (as a single layer neural network) - test accuracy of ~70% [notebook](https://github.com/uditgt/coursera_deeplearning_specialization/blob/main/1.1%20Building%20Logistic%20Regression%20as%20NN.ipynb)
2. Building Shallow neural network - test accuracy of ~72% [notebook](https://github.com/uditgt/coursera_deeplearning_specialization/blob/main/1.2%20Building%20Shallow%20NN%20using%20NumPy.ipynb)
3. Building Deep neural network - test accuracy of 82% [notebook](https://github.com/uditgt/coursera_deeplearning_specialization/blob/main/1.3%20Building%20Deep%20NN%20using%20NumPy.ipynb)

Focus of these exercises was on carefully piecing together the different equations for forward and backward pass, in vectorized form, to build a functioning neural network.

## Improving Deep Neural Networks
The course focuses on how to tune a neural networks, and discusses hyper-parameter tuning, regularization and optimization methods in more details. Including various approaches for **Regularization**, such as L2, drop-out, data augmentation; for **Optimization**, such as normalizing inputs, batch normalization, initializing neurons using Xavier method to manage variance to avoid vanishing/ exploding gradients; improving **Gradient Descent** algorithm by using mini-batches, Momentum / RMSprop / Adam methods, learning rate decay; more effectively tuning **Hyper-parameters** using random sampling approach (vs. grid search), prioritizing certain parameters over others, using appropriate scale for each hyper-parameter.

## Structuring Machine Learning Projects
The courses focuses on lots of practical do's and dont's when working on a ML project, including metric selection, structuring different datasets (train, train-dev, dev, test), analysing errors, addressing data mismatch, transfer learning, multi-task learning, structuring problems as end-to-end deep learning model vs. breaking into smaller parts. 

## Convolution Neural Networks
This course takes on whirlwind tour into the fastastic world of CNNs. Provides a foundational understanding manipulating 3D /volume data, how convolution and pooling operations work, what different layers mean and represent. After coding Forward and Backward propogation functions from scratch (using just numpy), we learn TensorFlow's - [Sequential API](https://www.tensorflow.org/guide/keras/sequential_model) and [Functional API](https://www.tensorflow.org/guide/keras/functional). 

Next we go over various architectures published over the years (and use some through transfer learning) - LeNet, AlexNet, VGG, ResNet, Inception Networks, MobileNet - learning various tricks developed in these networks. Object localization and detection tasks, including learning about bounding box, sliding window through convolution, IoU, non-max suppression, semantic segmentation using YOLO, U-Net architectures. Next we look at face recognition tasks using Siamese Network, DeepFace, FaceNet. Finally, we look at Neural Style Transfer architecture used for merging 'content' and 'style' images. 

## Sequence Models
In this course we learn about **RNN models** used when input and/or output data consists of a time 'sequence'. We look at **GRU & LSTM units** which have a memory cell and improve performance, by resolving the vanishing gradient problem and allowing for learning longer context/ relationships in data. Next we look at different architectures for RNN (1:1, 1:many, many:many etc.), including extensions such as **Bi-directional & Deep RNNs**.
Next we look at application of RNNs for **NLP** tasks, such as language modeling - sequence generation, learning word embedding (word2vec, GloVe), sentiment classification; machine translation (beam search algorithm, Bleu score, attention model, CTC cost function), trigger word detection with audio data.




## Interesting References:
* Computer Vision lecture [series](https://pjreddie.com/courses/computer-vision/) by Joseph Redmon, [youtube](https://www.youtube.com/playlist?list=PLjMXczUzEYcHvw5YYSU92WrY8IwhTuq7p) playlist
* Deeplearning Lectures [playlist](https://www.youtube.com/c/Deeplearningai/playlists)
* Dropout regularization, Gofrrey Hinton (2014) [link](https://jmlr.org/papers/v15/srivastava14a.html)
* Xavier initialization (2010) [link](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
