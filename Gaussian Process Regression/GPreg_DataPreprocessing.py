#preprocessing data for GP regressions

# --- READING TRAINING DATA --- #

import pandas as pd
import geopandas as gpd
import numpy as np

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

def reverse_tuple(x):
    '''function that reverses the order of a tuple'''
    y = (x[1], x[0])
    return(y)



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

# --- SAVING THE DATA --- #

np.savetxt('training_X.txt', training_X)
np.savetxt('y_label.txt', y)
np.savetxt('test_X.txt', test_X)


