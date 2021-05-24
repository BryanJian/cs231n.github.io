# %%
import numpy as np
from PIL import Image
from tensorflow.keras.datasets import cifar10

# %%
cat_img = Image.open("../assets/cat.jpg")
cat_arr = np.array(cat_img)
# %%
cf10 = cifar10.load_data()

# %%
