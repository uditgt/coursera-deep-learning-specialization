# [Coursera's Deep Learning specialization by Deeplearning.ai](https://www.coursera.org/specializations/deep-learning)

[Lecture Notes 📓](https://github.com/uditgt/coursera_deeplearning_specialization/tree/main/notes) | [Quizzes 📝](https://github.com/uditgt/coursera_deeplearning_specialization/tree/main/quizzes) | [Paper Review & Summaries 📌](https://github.com/uditgt/coursera_deeplearning_specialization/tree/main/papers) | [Certification 🎓](https://www.coursera.org/account/accomplishments/specialization/certificate/Y23QW2JU39ZE)

## Neural Networks & Deep Learning
The coursework provides a foundational understanding of a neural network. Different NN-based classifier models were built from scratch, using first principles - forward & backward propagation, calculating gradients and optimizing using gradient descent approach, **using mostly NumPy**. Classifiers built:
* Building Logistic regression classifier (as a single layer neural network) - test accuracy of ~70% [notebook 📃](https://github.com/uditgt/coursera_deeplearning_specialization/blob/main/1.1%20Building%20Logistic%20Regression%20as%20NN.ipynb)
* Building Shallow neural network - test accuracy of ~72% [notebook 📃](https://github.com/uditgt/coursera_deeplearning_specialization/blob/main/1.2%20Building%20Shallow%20NN%20using%20NumPy.ipynb)
* Building Deep neural network - test accuracy of 82% [notebook 📃](https://github.com/uditgt/coursera_deeplearning_specialization/blob/main/1.3%20Building%20Deep%20NN%20using%20NumPy.ipynb)

Focus of these exercises was on carefully piecing together the different equations for forward and backward pass, in vectorized form, to build a functioning neural network.

## Improving Deep Neural Networks
The course focuses on how to tune a neural networks, and discusses hyper-parameter tuning, regularization and optimization methods in more details. Including various approaches for **Regularization**, such as L2, drop-out, data augmentation; for **Optimization**, such as normalizing inputs, batch normalization, initializing neurons using Xavier method to manage variance to avoid vanishing/ exploding gradients; improving **Gradient Descent** algorithm by using mini-batches, Momentum / RMSprop / Adam methods, learning rate decay; more effectively tuning **Hyper-parameters** using random sampling approach (vs. grid search), prioritizing certain parameters over others, using appropriate scale for each hyper-parameter.

* Initializing parameters using zeros (creates symmetry problem), random, or Xavier method [notebook 📃](https://github.com/uditgt/coursera_deeplearning_specialization/blob/main/2.1%20Tuning%20Parameter%20Initialization.ipynb)
* Applying regularization through dropout method or L2 regularization through cost function [notebook 📃](https://github.com/uditgt/coursera_deeplearning_specialization/blob/main/2.2%20Regularization.ipynb)
* Calculation gradeint using numerical approximation methods [notebook 📃](https://github.com/uditgt/coursera_deeplearning_specialization/blob/main/2.3%20Gradient%20Checking.ipynb)
* Using diff Optimization methods - Gradient Descent, GD with momentum, Adaptive-moment (Adam) [notebook 📃](https://github.com/uditgt/coursera_deeplearning_specialization/blob/main/2.4%20Optimization%20Methods.ipynb)


## Structuring Machine Learning Projects
The courses focuses on lots of practical do's and dont's when working on a ML project, including metric selection, structuring different datasets (train, train-dev, dev, test), analysing errors, addressing data mismatch, transfer learning, multi-task learning, structuring problems as end-to-end deep learning model vs. breaking into smaller parts. 

## Convolution Neural Networks
This course takes on whirlwind tour into the fastastic world of CNNs. Provides a foundational understanding manipulating 3D /volume data, how convolution and pooling operations work, what different layers mean and represent. After coding Forward and Backward propogation functions from scratch (using just numpy), we learn TensorFlow's - Sequential and Functional API. 

Next we go over various architectures published over the years (and use some through transfer learning) - LeNet, AlexNet, VGG, ResNet, Inception Networks, MobileNet - learning various tricks developed in these networks. Object localization and detection tasks, including learning about bounding box, sliding window through convolution, IoU, non-max suppression, semantic segmentation using YOLO, U-Net architectures. Next we look at face recognition tasks using Siamese Network, DeepFace, FaceNet. Finally, we look at Neural Style Transfer architecture used for merging 'content' and 'style' images. 

* Tensorflow's Sequential / Function APIs - Happy Face dataset (65% accuracy) / Hand Sign dataset (78% accuracy) [notebook 📃](https://github.com/uditgt/coursera_deeplearning_specialization/blob/main/4.1%20CNN%20application%20using%20TensorFlow.ipynb)
* ResNet50 model - Hand Sign dataset (94% accuracy) [notebook 📃](https://github.com/uditgt/coursera_deeplearning_specialization/blob/main/4.2%20ResNet50%20on%20Hands%20dataset.ipynb)
* MobileNet on image dataset with augmentation (97% accuracy) [notebook 📃](https://github.com/uditgt/coursera_deeplearning_specialization/blob/main/4.3%20MobileNet%20transfer%20learning.ipynb)
* YOLO object detection [notebook 📃](https://github.com/uditgt/coursera_deeplearning_specialization/blob/main/4.4.%20YOLO%20Object%20Detection.ipynb)
* UNet image segmentation (97% accuracy) [notebook 📃](https://github.com/uditgt/coursera_deeplearning_specialization/blob/main/4.5%20UNet%20image%20segmentation.ipynb)
* FaceNet face recognition [notebook 📃](https://github.com/uditgt/coursera_deeplearning_specialization/blob/main/4.6%20FaceNet%20Face%20Recognition.ipynb)
* Neural Style Transfer [notebook 📃](https://github.com/uditgt/coursera_deeplearning_specialization/blob/main/4.7%20Neural%20Style%20Transfer.ipynb)

## Sequence Models
In this course we learn about **RNN models** used when input and/or output data consists of a time 'sequence'. We look at **GRU & LSTM units** which have a memory cell and improve performance, by resolving the vanishing gradient problem and allowing for learning longer context/ relationships in data. Next we look at different architectures for RNN (1:1, 1:many, many:many etc.), including extensions such as **Bi-directional & Deep RNNs**.
Next we look at application of RNNs for **NLP** tasks, such as language modeling - sequence generation, learning word embedding (word2vec, GloVe), sentiment classification; machine translation (beam search algorithm, Bleu score, attention model, CTC cost function), trigger word detection with audio data.

* Building a Language Model using just NumPy and making up Dinosaur names (Ychosaurus!) [notebook 📃](https://github.com/uditgt/coursera_deeplearning_specialization/blob/main/5.1%20Building%20Language%20model%20using%20NumPy%20(and%20making%20up%20Dinosaur%20names).ipynb)
* Audio sampling model (using LSTM) for generating Jazz music using Tensorflow [notebook 📃](https://github.com/uditgt/coursera_deeplearning_specialization/blob/main/5.2%20RNN%20Audio%20-%20training%20and%20sampling%20jazz%20music.ipynb)
* NLP - Exploring word Embeddings, Analogies and Gender-debiasing word vectors [notebook 📃](https://github.com/uditgt/coursera_deeplearning_specialization/blob/main/5.3%20NLP%20-%20Word%20Embeddings%20%26%20Debiasing.ipynb)
* NLP - Sentence classification to produce relevant emoji [notebook 📃](https://github.com/uditgt/coursera_deeplearning_specialization/blob/main/5.4%20NLP%20-%20Emojify.ipynb)
* NLP - Neural Machine Translation using Attention model [notebook 📃](https://github.com/uditgt/coursera_deeplearning_specialization/blob/main/5.5%20NLP%20-%20Neural%20Machine%20Translation.ipynb)
* NLP - Trigger word detection in audio data [notebook 📃](https://github.com/uditgt/coursera_deeplearning_specialization/blob/main/5.6%20NLP%20-%20Trigger%20word%20detection.ipynb)
* NLP - Introduction to Transformers [notebook 📃](https://github.com/uditgt/coursera_deeplearning_specialization/blob/main/5.7%20NLP%20-%20Transformers.ipynb)
* NLP - Transformers - Intuition behinb Positional Encoding [notebook 📃](https://github.com/uditgt/coursera-deep-learning-specialization/blob/main/5.8%20NLP%20-%20Transformers%20-%20Positional%20encoding%20intuition.ipynb)
* NLP - Transformers for Named Entity Recognition using [🤗 Huggingface](https://huggingface.co/) library [notebook 📃](https://github.com/uditgt/coursera-deep-learning-specialization/blob/main/5.9%20NLP%20-%20Transformers%20-%20Named%20entity%20recognition.ipynb)
* NLP - Transformers for Question / Answer task using [🤗 Huggingface](https://huggingface.co/) library [notebook 📃](https://github.com/uditgt/coursera-deep-learning-specialization/blob/main/5.10%20NLP%20-%20Transformers%20-%20Question%20Answering.ipynb)

## Interesting References:
* Computer Vision lecture [series](https://pjreddie.com/courses/computer-vision/) by Joseph Redmon, [🎥 playlist](https://www.youtube.com/playlist?list=PLjMXczUzEYcHvw5YYSU92WrY8IwhTuq7p) playlist
* Deeplearning Lectures [🎥 playlist](https://www.youtube.com/c/Deeplearningai/playlists)
* Yann LeCun's [website](http://yann.lecun.com/), [presentations](https://drive.google.com/drive/folders/0BxKBnD5y2M8NUXhZaXBCNXE4QlE?resourcekey=0-WtYv0wV-8DFNsFWfRUcpsw), [Deep Learning course @NYU](https://cds.nyu.edu/deep-learning/), [🎥 playlist](https://www.youtube.com/playlist?list=PLLHTzKZzVU9e6xUfG10TkTWApKSZCzuBI)
* 🤗 Huggingface: [`transformers` package](https://pypi.org/project/transformers/), [model summary](https://huggingface.co/docs/transformers/model_summary), 
* Some other sources (not reviewed): [understanding transformers](http://peterbloem.nl/blog/transformers)
* Stylish notes on Tess Fernandez [scribd](https://www.slideshare.net/TessFerrandez/notes-from-coursera-deep-learning-courses-by-andrew-ng), [website](https://www.tessferrandez.com/presentations/)
