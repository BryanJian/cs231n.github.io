import matplotlib.pyplot as plt

# %%
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.datasets import cifar10

sns.set()

# %%
(Xtr, Ytr), (Xte, Yte) = cifar10.load_data()
Image.fromarray(np.uint8(Xtr[0]))

# %%
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)  # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)  # Xte_rows becomes 10000 x 3072

# %% [markdown]
# class NearestNeighbor(object):
#     def __init__(self):
#         pass

#     def train(self, X, y):
#         """ X is N x D where each row is an example. Y is 1-dimension of size N """
#         the nearest neighbor classifier simply remembers all the training data
#         self.Xtr = X
#         self.ytr = y


#     def predict(self, X):
#         """ X is N x D where each row is an example we wish to predict label for """
#         num_test = X.shape[0]
#         lets make sure that the output type matches the input type
#         Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

#         loop over all test rows
#         for i in range(num_test):
#             find the nearest training image to the i'th test image
#             using the L1 distance (sum of absolute value differences)
#             distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
#             min_index = np.argmin(distances)  # get the index with smallest distance
#             Ypred[i] = self.ytr[min_index]  # predict the label of the nearest example

#         return Ypred


# %% [markdown]
# nn = NearestNeighbor()  # create a Nearest Neighbor classifier class
# nn.train(Xtr_rows, Ytr)  # train the classifier on the training images and labels
# Yte_predict = nn.predict(Xte_rows)  # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
# print("accuracy NN: %f" % (np.mean(Yte_predict == Yte)))

# %%
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Xtr_rows, Ytr.ravel())
Yte_pred = knn.predict(Xte_rows)
print("accuracy KNN: %f" % (np.mean(Yte_pred == Yte.ravel())))
# %%
# assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
# recall Xtr_rows is 50,000 x 3072 matrix
Xval_rows = Xtr_rows[:1000, :]  # take first 1000 for validation
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:, :]  # keep last 49,000 for train
Ytr = Ytr[1000:]

# %%
# find hyperparameters that work best on the validation set
validation_accuracies = []
for k in [1, 3, 5, 7, 10, 20, 50, 100]:

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xtr_rows, Ytr.ravel())
    Yval_pred = knn.predict(Xval_rows)
    acc = np.mean(Yval_pred == Yval.ravel())
    print("accuracy of k = {}: {}".format(k, acc))

    # keep track of what works on the validation set
    validation_accuracies.append((k, acc))

_ = plt.plot(*zip(*validation_accuracies))
_ = plt.xlabel("k")
_ = plt.ylabel("Cross-validation accuracy")
# %%
