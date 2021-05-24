# %%

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

# %%
a = np.array([1, 2, 3])
b = np.array([[1, 2, 3], [4, 5, 6]])
c = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
e = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
f = np.array(
    [
        [[1, 2, 3], [9, 8, 8], [4, 5, 6]],
        [[1, 2, 3], [4, 5, 6], [9, 8, 8]],
        [[1, 2, 3], [4, 5, 6], [9, 8, 8]],
    ]
)

# %%
# List comprehension + flattening array
[x for x in e.flatten() if (x % 2) != 0]

# %%
e[e % 2 != 0]
# %%
# Integer array indexing
# First bracket picks elements in first rank
c1 = c[[0, 1, 2], [0, 0, 0]]

# %%
# SciPy
# Read an JPEG image into a numpy array
img = Image.open("../assets/cat.jpg")
img_arr = np.array(img)
img_arr = img_arr * [1, 0.95, 0.9]
img_tinted = Image.fromarray(np.uint8(img_arr))
img_resize = img_tinted.resize((300, 300))

# %%
x = np.array([[0, 1], [1, 0], [2, 0]])

plt.scatter(*zip(*x))

d = squareform(pdist(x, "euclidean"))
# %%
# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(3, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title("Sine")

# Set the second subplot as active, and make the second plot.
plt.subplot(3, 1, 2)
plt.plot(x, y_cos)
plt.title("Cosine")

plt.subplot(3, 1, 3)
plt.imshow(np.uint8(img_arr))

# Show the figure.
plt.show()
# %%
