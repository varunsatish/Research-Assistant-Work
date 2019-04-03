#basic gaussian process regression with a Euclidean Distance measure



# --- CONSTRUCTING KERNEL FUNCTIONS --- #


import gpflow
import tensorflow as tf


class SE_Euclid(gpflow.kernels.Kernel):
    """
    Squared Exponential Kernel utilising Euclidean Distance
    """

#Note: Parameters need to be tensors

    def __init__(self, variance, lengthscale):
        super().__init__(input_dim=1, active_dims=[0])
        self.variance = gpflow.Param(1.0, transform=gpflow.transforms.positive)
        self.lengthscale = gpflow.Param(1.0, transform=gpflow.transforms.positive)

    def square_dist(self, X, X2):
        X = X / self.lengthscales
        Xs = tf.reduce_sum(tf.square(X), axis=1)

        if X2 is None:
            dist = -2 * tf.matmul(X, X, transpose_b=True)
            dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
            return (dist)

        X2 = X2 / self.lengthscales
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
        self.variance * tf.exp(- self.euclid_dist(X, X2) * (1 / self.lengthscale ** 2))


# --- READING TRAINING DATA --- #

import pandas as pd
import geopandas as gpd

filenames = ['2016Census_G02_AUS_SA1.csv',
             'MB_2016_NSW.csv', 'SA1_2016_AUST.shp']

file_paths = ['/Users/varunsatish/CTDS_Data/2016_Data_CTDS/SA1/AUST/',
              '/Users/varunsatish/CTDS_Data/',
              '/Users/varunsatish/spatial_project/geography_files/']

df = pd.read_csv(file_paths[0] + filenames[0])

meshblocks = pd.read_csv(file_paths[1] + filenames[1])

#importing the shapefiles
geometries = gpd.read_file(file_paths[2] + filenames[2])

#adding geometry column to df
df['geometry'] = geometries['geometry']

df = gpd.GeoDataFrame(df)

#dropping rows of df wil 'None' statistics or polygons
df = df.dropna()

from shapely.geometry import Point

#finding the centroids of meshblocks
df['centroids'] = df['geometry'].centroid

def SA2_to_SA1(SA2code):
    '''returns the SA1 codes that aggregate to an SA2 region'''
    df = meshblocks[meshblocks.SA2_NAME_2016 == SA2code]
    # we must use unique since the dataframe contains codes relating to meshblocks which are smaller
    y = list(df.SA1_7DIGITCODE_2016.unique())
    return (y)


def GSYD(code):
    '''returns the SA1 codes that aggregate to a GCCSA Code'''
    df = meshblocks[meshblocks.GCCSA_NAME_2016 == code]
    # we must use unique since the dataframe contains codes relating to meshblocks which are smaller
    y = list(df.SA1_7DIGITCODE_2016.unique())
    return (y)


G_list = ['Greater Sydney']

inner_west_list = ['Marrickville',
                   'Petersham - Stanmore',
                   'Erskinevile - Alexandria',
                   'Glebe - Forrest Lodge',
                   'Newtown - Camperdown - Darlington',
                   'Redfern - Chippendale']

# returns a list of lists
sa1 = [SA2_to_SA1(x) for x in inner_west_list]
G = [GSYD(x) for x in G_list]

# returns a 'flat' list, useful when len(inner_west_list) > 1
sa1 = [str(item) for sublist in sa1 for item in sublist]
G = [str(item) for sublist in G for item in sublist]

# for each item in sa1 we want to obtain latittudes/longtitudes and centroids

inner_west = df[df['SA1_7DIGITCODE_2016'].isin(G)]

# isolating centroids
training_X = inner_west['centroids']

# obtaining coordinates
training_X = [x.coords[0] for x in training_X]

# reversing to properly format
training_X = [reverse_tuple(x) for x in training_X]

# splitting into lattitude and longtitudes
training_X1 = [x[0] for x in training_X]
training_X2 = [x[1] for x in training_X]

# creating into an appropriate array
training_X = np.array(training_X).reshape(len(training_X), 2)

# obtaining labels
y = np.array([np.float(x) for x in inner_west['Median_rent_weekly']])
y = y.reshape(len(y), 1)


# --- CONSTRUCTING TEST DATA --- #

import numpy as np

#constructing a grid of test points
#the grid is relative to the mean of training points

x1 = np.linspace(np.mean(training_X1) + 0.5, np.mean(training_X1) - 0.5, num = 5) #x
x2 = np.linspace(np.mean(training_X2) - 0.5 ,  np.mean(training_X2) + 0.5,num = 5) #y
xx1, xx2 = np.meshgrid(x1,x2)
test_X = np.vstack((xx1.flatten(),xx2.flatten())).T


# --- CREATING THE GP MODEL --- #

#creating a GP model with fixed parameters

k = RBF_manhattan(1,1)
m = gpflow.models.GPR(training_X, y, kern=k)

#running the model
mean, var = m.predict_f(Xplot)

#contour plot with Newtown Centroid for reference

plt.contour(xx1, xx2, np.array(mean).reshape(xx1.shape))
plt.xlim([np.mean(training_X1) + 0.5, np.mean(training_X1) - 0.5])
plt.ylim([np.mean(training_X2) - 0.5, np.mean(training_X2) + 0.5])
plt.plot(-33.8970,151.1793,'rx')
plt.text(-33.8970 - 0.003, 151.1793 + 0.03, 'Newtown', fontsize=12)







