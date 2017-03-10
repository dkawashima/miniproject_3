import numpy as np
import matplotlib.pyplot as plt
# Basic Visualization for movie data

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

# 4.1 All ratings in MovieLens DataSet
'''
plt.hist(dataList[:, 2], bins=np.arange(1, 7) -0.5)
plt.title("MovieLens Ratings Histogram")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.savefig('allRatings.jpg')
'''


movie_to_ratings = {}
movie_to_count = {}

for elem in dataList:
	if int(elem[1]) not in movie_to_ratings.keys():
		movie_to_ratings[int(elem[1])] = [elem[2]]
		movie_to_count[int(elem[1])] = 1
	else:
		movie_to_ratings[int(elem[1])].append(elem[2])
		movie_to_count[int(elem[1])] += 1


# 4.2 Visualize ratings of top ten most popular movies
ten_most_pop = []
avg_ratings_most_pop = []
# Find top ten movies with highest number of ratings
for movie in sorted(movie_to_count.items(), key=lambda x: x[1], reverse=True)[:10]:
	ten_most_pop.append(movie[0])

print ('Top Ten Most Popular: ')
for m in ten_most_pop:
	print ('Movie Name: ', movieList[m - 1][1])
	avg_ratings_most_pop.append(np.mean(movie_to_ratings[m]))

plt.boxplot(avg_ratings_most_pop, vert=False)
plt.title("Boxplot of Average Ratings of Top Ten Most Popular Movies")
plt.xlabel("Rating")
plt.savefig('topTenPopRatings.jpg')
plt.close()

# 4.3 Visualize ratings of top ten best movies
ten_best = []
ten_best_5 = []
# Add threshold to filter out movies with 5 or less ratings
avg_ratings_best = []
avg_ratings_best_5 = []
# Find top ten movies with highest average rating
for movie in sorted(movie_to_ratings.items(), key=lambda x: np.mean(x[1]), reverse=True):
	ten_best.append(movie[0])
	if (len(movie[1]) > 5):
		ten_best_5.append(movie[0])

print ('Top Ten Best, No Threshold: ')
for m in ten_best[:10]:
	print ('Movie Name: ', movieList[m - 1][1])
	avg_ratings_best.append(np.mean(movie_to_ratings[m]))

print ('Top Ten Best, 5 Reviews or More: ')
for m in ten_best_5[:10]:
	print ('Movie Name: ', movieList[m - 1][1])
	avg_ratings_best_5.append(np.mean(movie_to_ratings[m]))



plt.hist(avg_ratings_best, bins=np.arange(1, 7) -0.5)
plt.title("Histogram of Average Rating of Top Ten Best Movies")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.savefig('topTenBestRatings.jpg')
plt.close()

plt.boxplot(avg_ratings_best_5, vert=False)
plt.title("Boxplot of Average Rating of Top Ten Best Movies")
plt.xlabel("Rating")
plt.savefig('topTenBestRatings_5.jpg')
plt.close()

# 4.4 Visualize ratings of three genres: Comedy, Action, and Thriller
actionRatings = []
comedyRatings = []
thrillerRatings = []

countAction = 0
countComedy = 0
countThriller = 0

for movie in movieList:
	if movie[3] == 1: # Action movie
		actionRatings = actionRatings + movie_to_ratings[movie[0]]
		countAction += 1
	if movie[7] == 1: # Comedy movie
		comedyRatings = comedyRatings + movie_to_ratings[movie[0]]
		countComedy += 1
	if movie[18] == 1: # Thriller movie
		thrillerRatings = thrillerRatings + movie_to_ratings[movie[0]]
		countThriller += 1

print ("Number of Action Movies: ", countAction)
print ("Number of Comedy Movies: ", countComedy)
print ("Number of Thriller Movies: ", countThriller)


plt.hist(actionRatings, bins=np.arange(1, 7) -0.5)
plt.title("Histogram of Ratings of All Action Movies")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.savefig('actionRatings.jpg')
plt.close()

plt.hist(comedyRatings, bins=np.arange(1, 7) -0.5)
plt.title("Histogram of Ratings of All Comedy Movies")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.savefig('comedyRatings.jpg')
plt.close()

plt.hist(thrillerRatings, bins=np.arange(1, 7) -0.5)
plt.title("Histogram of Ratings of All Thriller Movies")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.savefig('thrillerRatings.jpg')

