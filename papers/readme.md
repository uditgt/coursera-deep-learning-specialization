### Remarkable Ideas

[Neural Style Transfer (Gatys, 2015)](https://github.com/uditgt/DeepLearning/blob/main/papers/Paper%20-%20Neural%20Style%20Transfer%20(Gatys%2C%202015).pdf)
  * Combines 'content' and 'style' from two images. Cost function with two components - content and style. 
  * Content is captured in higher layers of the network (VGG used; with max-pool changed to avg-pool for better gradient flow), while Style is captured throughout the network.
  * Style is calculated as 'Gram matrix', which is basically element-wise sumproduct between (two at a time) channels of a layer
  * Start with a white noise image, and modify it using gradient descent to minimize the cost function.
 
  <p align="center"><img width="400" height="200" src="https://github.com/uditgt/DeepLearning/blob/main/papers/image_gatys.png"></p>
  
