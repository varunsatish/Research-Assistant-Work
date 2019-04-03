#basic gaussian process regression with a Euclidean Distance measure

# ---   IMPORTING RELEVANT DATA --- #
import numpy as np

filenames = ['training_X.txt', 'y_label.txt', 'test_X']
directory = '/Users/varunsatish/PycharmProjects/GPreg/'

training_X = np.loadtxt(directory + filenames[0])
y = np.loadtxt(directory + filenames[1])
test_X = np.loadtxt(directory + filenames[2])

# --- CONSTRUCTING KERNEL FUNCTIONS --- #


import gpflow
import tensorflow as tf


import tensorflow as tf


class SE_Euclid(gpflow.kernels.Kernel):
    """
    Squared Exponential Kernel utilising Euclidean Distance
    """

#Note: Parameters need to be tensors

    def __init__(self, variance, lengthscale):
        super().__init__(input_dim= 2)
        self.variance = gpflow.Param(1.0, transform=gpflow.transforms.positive)
        self.lengthscale = gpflow.Param(1.0, transform=gpflow.transforms.positive)

    def square_dist(self, X, X2):
        X = X / self.lengthscale
        Xs = tf.reduce_sum(tf.square(X), axis=1)

        if X2 is None:
            dist = -2 * tf.matmul(X, X, transpose_b=True)
            dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
            return (dist)

        X2 = X2 / self.lengthscale
        X2s = tf.reduce_sum(tf.square(X2), axis=1)
        dist = -2 * tf.matmul(X, X2, transpose_b=True)
        dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))
        return (dist)

    def euclid_dist(self, X, X2):
        r2 = self.square_dist(X, X2)
        return (tf.sqrt(r2 + 1e-12))

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
            return (self.variance * tf.exp(- self.square_dist(X, X2) * (1 / self.lengthscale ** 2)))
    def Kdiag(self, X, presliced=False):
        return(tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance)))

# --- CREATING THE GP MODEL --- #

#creating a GP model with fixed parameters

k = SE_Euclid(1,1)
m = gpflow.models.GPR(training_X, y, kern=k)

#running the model
mean, var = m.predict_y(test_X)

#contour plot with Newtown Centroid for reference

import matplotlib.pyplot as plt

plt.contour(xx1, xx2, np.array(mean).reshape(xx1.shape))
plt.xlim([np.mean(training_X1) + 0.5, np.mean(training_X1) - 0.5])
plt.ylim([np.mean(training_X2) - 0.5, np.mean(training_X2) + 0.5])
plt.plot(-33.8970,151.1793,'rx')
plt.text(-33.8970 - 0.003, 151.1793 + 0.03, 'Newtown', fontsize=12)
plt.show()
