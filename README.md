# scikit-learn
Handwritten digit recognition with scikit-learn
* We trained a simple neural network to recognize the numbers in these images. This network will take 1D arrays of 8x8=64 values as input. We then converted these 2D images into 1D arrays

<hr>

### We start by loading the sample
```python
from sklearn import datasets
digits = datasets.load_digits()
```
### Then we print the first image
```python
print(digits.images[0])
```
```terminal
[[ 0.  0.  5. 13.  9.  1.  0.  0.]
 [ 0.  0. 13. 15. 10. 15.  5.  0.]
 [ 0.  3. 15.  2.  0. 11.  8.  0.]
 [ 0.  4. 12.  0.  0.  8.  8.  0.]
 [ 0.  5.  8.  0.  0.  9.  8.  0.]
 [ 0.  4. 11.  0.  1. 12.  7.  0.]
 [ 0.  2. 14.  5. 10. 12.  0.  0.]
 [ 0.  0.  6. 13. 10.  0.  0.  0.]]
 ```
 ### Like all the images in the sample, this one is an 8x8 pixel image, black and white (a single color level per pixel). It can be displayed in the following way, also indicating the corresponding label (the number to which the image corresponds)
 ```python
 import matplotlib.pyplot as plt
plt.imshow(digits.images[0],cmap='binary')
plt.title(digits.target[0])
plt.axis('off')
plt.show()
```
<img src="output1.png" width="80%">
