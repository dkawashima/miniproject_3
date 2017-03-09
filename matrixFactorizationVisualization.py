# Ricky Galliani, David Kawashima, Eshan Govil
# CS/CNS/CMS/EE 155
# Matrix Factorization Visualizations

import numpy as np
import matplotlib.pyplot as plt
import prob2utils

# Read in the data

f_data = open('./data.txt')
f_movies = open('./movies.txt')

dataLines = f_data.readlines()
moviesLines = f_movies.readlines()

dataList = np.zeros((100000, 3))
for i, line in enumerate(dataLines):
    elems = line.split('\t')
    dataList[i][0] = int(elems[0])
    dataList[i][1] = int(elems[1])
    dataList[i][2] = int(elems[2])

movieList = []
for i, line in enumerate(moviesLines):
    elems = line.split('\t')
    tempList = []
    tempList.append(int(elems[0]))
    tempList.append(elems[1])
    for j in range(2, 19):
        tempList.append(int(elems[j]))
    movieList.append(tempList)

# Y is m x n matrix of movie ratings, where y_ij is user i's 
# rating for movie j
Y = np.zeros((943, len(movieList)))

# Note: movie's # is 1 + movieIndex in Y
# Note: user's # is 1 + userIndex in Y
for i in range(len(dataList)):
    Y[int(dataList[i][0]) - 1][int(dataList[i][1]) - 1] = int(dataList[i][2])

# print (Y)

