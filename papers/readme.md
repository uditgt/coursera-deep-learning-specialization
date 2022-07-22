### Key Ideas

[Neural Style Transfer (Gatys, 2015)](https://github.com/uditgt/DeepLearning/blob/main/papers/Paper%20-%20Neural%20Style%20Transfer%20(Gatys%2C%202015).pdf)
  * Combines 'content' and 'style' from two images. Cost function with two components - content and style. 
  * Content is captured in higher layers of the network (VGG used in the paper with max-pool changed to avg-pool), while Style is captured in throughout the network, and for a smooth visual flow, authors use style information from all the layers in the network.
  * Operationally - start with a white noise image, and modify it using gradient descent to minimize the defined cost function.
 
  <p align="center"><img width="400" height="200" src="https://github.com/uditgt/DeepLearning/blob/main/papers/image_gatys.png"></p>
  
