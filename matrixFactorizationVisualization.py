# Ricky Galliani, David Kawashima, Eshan Govil
# CS/CNS/CMS/EE 155
# Matrix Factorization Visualizations

import sys
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from prob2utils import *

movie_cols = ['Movie ID','Movie Title', 'Unkown', 'Action', 'Adventure', \
              'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', \
              'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', \
              'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('movies.txt', delimiter='\t', names=movie_cols)
print('Read in movie data...')

ratings_cols = ['User ID', 'Movie ID', 'Rating']
ratings = pd.read_csv('data.txt', delimiter='\t', names=ratings_cols)
print('Read in ratings data...')

# Y is M x N matrix of movie ratings, where y_ij is user i's 
# rating for movie j
M = 943 # Users
N = 1682 # Movies
Y = []

# Note: movie's # is 1 + movieIndex in Y
# Note: user's # is 1 + userIndex in Y
numMovieRatings = {}
avgMovieRatings = {}
comedyFilms = {}
thrillerFilms = {}
actionFilms = {}
for i, row in ratings.iterrows():
    movieID = int(row['Movie ID'])
    userID = int(row['User ID'])
    rating = int(row['Rating'])
    isComedy = int(movies.iloc[movieID - 1]['Comedy']) == 1
    isAction = int(movies.iloc[movieID - 1]['Action']) == 1
    isThriller = int(movies.iloc[movieID - 1]['Thriller']) == 1
    if isComedy:
        comedyFilms[movieID] = isComedy
    if isAction:
        actionFilms[movieID] = isAction
    if isThriller:
        thrillerFilms[movieID] = isThriller
    Y.append((userID, movieID, rating))
    if movieID not in numMovieRatings:
        numMovieRatings[movieID] = 0
        avgMovieRatings[movieID] = rating, 1
    else: 
        numMovieRatings[movieID] += 1
        existingRating = avgMovieRatings[movieID][0]
        numRatings = avgMovieRatings[movieID][1]
        newRating = ((existingRating * numRatings) + rating) / (numRatings + 1)
        avgMovieRatings[movieID] = (newRating, numRatings + 1)

Y = np.array(Y) # Make Y a numpy array
print('Built out Y matrix...')

def plotMovies(title, Vx, Vy, labels):
    '''
    Plots the input Vx and Vy.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(title)
    ax.scatter(Vx, Vy)
    for (x, y, label) in zip(Vx, Vy, labels):
        label = label[:-7]
        if ',' in label: 
            commaInd = label.index(',')
            postComma = label[commaInd + 2:]
            preComma = label[:commaInd]
            label = postComma + ' ' + preComma
        ax.annotate(label, (x + 0.03, y), fontsize=9)
    ax.set_xlim([min(Vx) - .1, max(Vx) + 1])
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    plt.show()

if __name__ == '__main__': 
    
    if len(sys.argv) != 6: 
        params = '<K> <eta> <reg> <eps> <epochs>'
        u = 'python matrixFactorizationVisualization.py ' + params
        print('[USAGE]: ' + u)
        exit(1)

    # Process command-line arguments
    K = int(sys.argv[1])
    eta = float(sys.argv[2])
    reg = float(sys.argv[3])
    eps = float(sys.argv[4])
    epochs = int(sys.argv[5])

    # Train the model using the Homework 6 solutions
    print('Training the model...')
    U, V, err = train_model(M, N, K, eta, reg, Y, eps=eps, max_epochs=epochs)
    print('Training completed...')

    A, sigma, B = np.linalg.svd(V)
    print('Computed SVD of V matrix...')

    At = A[:,:2]

    U = np.dot(U, At)
    print('Projected U to 2 dimensions...')
    V = np.dot(np.transpose(V), At)
    print('Projected V to 2 dimensions...')

    # Plot 10 random movies
    random10 = random.sample(range(N), 10)
    randomLabels = [row['Movie Title'] for i, row in movies.iterrows() \
                    if row['Movie ID'] in random10]
    Vx = [x for i, (x,y) in enumerate(V) if (i + 1) in random10]
    Vy = [y for i, (x,y) in enumerate(V) if (i + 1) in random10]
    print('Unpacked x and y coordinates from V matrix for random 10 plot...')
    plotMovies('10 Random Movies', Vx, Vy, randomLabels)

    # Plot the ten most popular movies
    popular10 = [x for (x,y) in \
        sorted(numMovieRatings.items(), key=lambda x:x[1], reverse=True)[:10]]
    popularLabels = [row['Movie Title'] for i, row in movies.iterrows() \
                    if row['Movie ID'] in popular10]
    print(str([movies.iloc[i + 1]['Movie Title'] for i in popular10]))
    Vx = [x for i, (x,y) in enumerate(V) if (i + 1) in popular10]
    Vy = [y for i, (x,y) in enumerate(V) if (i + 1) in popular10]
    print('Unpacked x and y coordinates from V matrix for popular 10 plot...')
    plotMovies('10 Most Popular Movies', Vx, Vy, popularLabels)

    # Plot the ten highest rated movies
    highest10 = [x for (x,y) in \
        sorted(avgMovieRatings.items(), key=lambda x:x[1][0], reverse=True)[:10]]
    highestLabels = [row['Movie Title'] for i, row in movies.iterrows() \
                    if row['Movie ID'] in highest10]
    Vx = [x for i, (x,y) in enumerate(V) if (i + 1) in highest10]
    Vy = [y for i, (x,y) in enumerate(V) if (i + 1) in highest10]
    print('Unpacked x and y coordinates from V matrix for highest rated 10 plot...')
    plotMovies('10 Highest Rated Movies', Vx, Vy, highestLabels)

    # Plot ten movies from action, thriller, and comedy
    comedy10 = [x for (x,y) in random.sample(comedyFilms.items(), 10)]
    thriller10 = [x for (x,y) in random.sample(thrillerFilms.items(), 10)]
    action10 = [x for (x,y) in random.sample(actionFilms.items(), 10)]
    allFilms = comedy10 + thriller10 + action10
    allFilmLabels = [row['Movie Title'] for i, row in movies.iterrows() \
                    if row['Movie ID'] in allFilms]
    Vx = [x for i, (x,y) in enumerate(V) if (i + 1) in allFilms]
    Vy = [y for i, (x,y) in enumerate(V) if (i + 1) in allFilms]
    print('Unpacked x and y coordinates from V matrix for three generes plot...')
    plotMovies('10 Action, 10 Thriller, and 10 Comedy Movies', Vx, Vy, allFilmLabels)


