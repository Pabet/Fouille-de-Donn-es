#%%
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
import itertools

def most_similar(tfidf, item_index, k):
    """Returns indices of k most similar items to item #item_index (excl. itself)

  Args:
    tfidf: a sparse matrix as returned by TfidfVectorizer.fit_transform(..)
    item_index: an integer, the index of the item (0-based). Eg. a row index.
    k: an integer, the number of similar item indices to return.

  Returns:
    A list of integers representing the items (eg. rows) most similar to item
    #item_index, excluding itself. Use the cosine similarity.
    """
    similarity_vector = []
    x_vector = tfidf[item_index, :]
    print(tfidf.shape[0], tfidf.shape[1])
    print(x_vector.shape[0], x_vector.shape[1])

    #calculate the similarities between all rows != x (item_index) 
    for y in range(0,tfidf.shape[0]):
        similarity_numerator = 0
        similarity_denominator = 0

        if(y != item_index):

            y_vector = tfidf[y, :]
            #for vector y iterate over words i
            #calculate cosine_similarity
            for i in range(0,tfidf.shape[1]):
                print(y, i)
                similarity_numerator += x_vector[i]*y_vector[i]
                similarity_denominator += x_vector[i]**2 + y_vector[i]**2

        similarity_vector.append(similarity_numerator / np.sqrt(similarity_denominator))


    #find the k most similar rows
    k_similarity_vector = []
    similarity_vector.sort()
    for i in range(k):
        k_similarity_vector.append(similarity_vector[i])

    return k_similarity_vector    




#data = open('jester_jokes.txt', 'r').readlines()
#vectorizer = TfidfVectorizer()
#X = vectorizer.fit_transform(data)


#print(most_similar(X, 0, 1))

#%%
from collections import defaultdict

def read_ratings(filename, num_jokes):
    """Parses a file in the same format as jester_ratings.csv.


  Args:
    filename: a string: the name of the file to parse.
    num_jokes: the global number of jokes (some may never appear in the file,
               Which is why this argument is necessary).

  Returns:
    A list of dictionaries: the list of ratings for each user. The ratings for a
    user are a dictionary {joke_id: rating}, where joke_id is an integer in
    0..num_jokes-1 and rating is a float in [-10,10].
    When a joke is not rated by a user, it should not be in the user's dictionary.
    """

    dict_list = []

 
    d = defaultdict(float)
    previous_user = 0

    for line in open(filename, 'r').readlines():
        line_list = line.split(',')
    
        if(int(line_list[0]) > int(previous_user)):
           
            dict_list.append(d)
            previous_user = int(line_list[0])
            d = defaultdict(float)     
        else:
            k = int(line_list[1])
            v = float(line_list[2])
            d[k] = v
        
    return dict_list

#l = read_ratings('jester_ratings.csv', 150)
#print(l[0])
#print(sum(l[0][x] for x in l[0]))

def content_recommend(similarity_matrix, user_ratings, k):
    """Recommends k best jokes for a given user.

  This recommendation takes as input the ratings of a single user, but only
  takes into account the ratings of even-numbered jokes, while it only recommends
  Odd-numbered jokes.

  Args:
    similarity_matrix: A similarity matrix of size NxN.
    user_ratings: a dictionary {joke id: rating} containing the known joke
                  ratings of a given user.
    k: an integer, the number of odd-indexed jokes to recommend.

  Returns:
    A list of odd joke indices recommended for this user, based on the joke
    similarities and using only the user's ratings of even-indexed jokes.
    """

    
