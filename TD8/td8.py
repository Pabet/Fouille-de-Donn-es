#%%
import numpy as np


def read_data(filename):
    """Reads a breast-cancer-diagnostic dataset, like wdbc.data.

  Args:
    filename: a string, the name of the input file.

  Returns:
    A pair (X, Y) of lists:
    - X is a list of N points: each point is a list of numbers
      corresponding to the data in the file describing a sample.
      N is the number of data points in the file (eg. lines).
    - Y is a list of N booleans: Element #i is True if the data point
      #i described in X[i] is "cancerous", and False if "Benign".
    """

    n_points = []
    n_booleans = []

    for line in open(filename, 'r').readlines():
        tmp = line[:-1]
        line_list = tmp.split(',')

        if(line_list[1] == 'M'):
            n_booleans.append(True)
        else:
            n_booleans.append(False)

        line_list = line_list[2:]    

        data_entry = []
        for value in line_list:
          data_entry.append(float(value))

        n_points.append(data_entry)       

    return (n_points, n_booleans)

#print(read_data('tmp.txt'))


def simple_distance(data1, data2):
    """Computes the Euclidian distance between data1 and data2.
  
  Args:
    data1: a list of numbers: the coordinates of the first vector.
    data2: a list of numbers: the coordinates of the second vector (same length as data1).

  Returns:
    The Euclidian distance: sqrt(sum((data1[i]-data2[i])^2)).
    """

    vector_sum = 0

    for i,j in zip(data1, data2):
        vector_sum += (i-j)**2

    return np.sqrt(vector_sum)    


#print(simple_distance([1.0, 0.4, -0.3, 0.15], [0.1, 4.2, 0.0, -1]))

def k_nearest_neighbors(x, points, dist_function, k):
    """Returns the indices of the k elements of "points" that are closest to "x".
    
  Args:
    x: a list of numbers: a N-dimensional vector.
    points: a list of list of numbers: a list of N-dimensional vectors.
    dist_function: a function taking two N-dimensional vectors as
       arguments and returning a number. Just like simple_distance.
    k: an integer. Must be smaller or equal to the length of "points".

  Returns:
    A list of integers: the indices of the k elements of "points" that are
    closest to "x" according to the distance function dist_function.
    IMPORTANT:
    """
    distance_pairs = []

    for i in range(0, len(points)):
        distance = dist_function(x, points[i])
        distance_pairs.append((i, distance))

    distance_pairs.sort(key=lambda  tup: tup[1])

    k_nearest = []

    for i in range(0, k):
        k_nearest.append(distance_pairs[i][0])

    return k_nearest

k_nearest_neighbors([1.2, -0.3, 3.4],
                   [[2.3, 1.0, 0.5], [1.1, 3.2, 0.9], [0.2, 0.1, 0.23], [4.1, 1.9, 4.0]],
                    simple_distance, 2)


import math
import random

def split_lines(input, seed, output1, output2):
  """Distributes the lines of 'input' to 'output1' and 'output2' pseudo-randomly.

  Args:
    input: a string, the name of the input file.
    seed: an integer, the seed of the pseudo-random generator used. The split
        should be different with different seeds. Conversely, using the same
        seed and the same input should yield exactly the same outputs.
    output1: a string, the name of the first output file.
    output2: a string, the name of the second output file.
  """
  random.seed(seed)
  out1 = open(output1, 'w')
  out2 = open(output2, 'w')
  for line in open(input, 'r').readlines():
    if random.randint(0, 1):
      out1.write(line)
    else:
      out2.write(line)



def is_cancerous_knn(x, train_x, train_y, dist_function, k):
    """Predicts whether some cells appear to be cancerous or not, using KNN.
    
  Args:
    x: A list of floats representing a data point (in the cancer dataset,
       that's 30 floats) that we want to diagnose.
    train_x: A list of list of floats representing the data points of
       the training set.
    train_y: A list of booleans representing the classification of
       the training set: True if the corresponding data point is
       cancerous, False if benign. Same length as 'train_x'.
    dist_function: A function taking two N-dimensional vectors as
       arguments and returning a number. Just like simple_distance.
    k: Same as in k_nearest_neighbors().

  Returns:
    A boolean: True if the data point x is predicted to be cancerous, False
        if it is predicted to be benign.
    """

   
   
    k_nearest = k_nearest_neighbors(x, train_x, dist_function, k)

    average = 0.0

    for i in k_nearest:
        if (train_y[i] == True):
            average += 1

    average /= len(k_nearest)

    if (average == 0.5):
            return train_y[k_nearest[0]]
    elif (average > 0.5):
            return True
    else:
        return False        

is_cancerous_knn([1.2, -0.3, 3.4],
                 [[2.3, 1.0, 0.5], [1.1, 3.2, 0.9], [0.2, 0.1, 0.23], [4.1, 1.9, 4.0]],
                 [False, False, True, False], simple_distance, 2)



def eval_cancer_classifier(test_x, test_y, classifier):
    """Evaluates a cancer KNN classifier.

  This takes an already-trained classifier function, and a test dataset, and evaluates
  the classifier on that test dataset: it calls the classifier function for each x in
  test_x, compares the result to the corresponding expected result in test_y, and
  computes the average error.
  
  Args:
    test_x: A list of lists of floats: the test/validation data points.
    test_y: A list of booleans: the test/validation data class (True = cancerous,
       False = benign)
    classifier: A classifier, i.e. a function whose sole argument is of the same
       Type as an element of train_x or test_x, and whose return value is
       The same type as train_y or test_y. For example:
       lambda x: is_cancerous_knn(x, train_x, train_y, dist_function=simple_distance, k=5)

  Returns:
    A float: the error rate of the classifier on the test dataset. This is
    a value in [0,1]: 0 means no error (we got it all correctly), 1 means
    we made a mistake every time. Note that choosing randomly yields an error
    rate of about 0.5, assuming that the values in test_y are all Boolean.
    """

    average = 0.0

    for i in range(0, len(test_x)):
        if(classifier(test_x[i]) == test_y[i]):
            average += 1.0

    average /= len(test_x)

    return average


split_lines('/home/pabet/Documents/s1/Fouillle de Données/TD8/wdbc.data', 5, 'train.txt', 'test.txt')
#split_lines('tmp.txt', 5, 'train.txt', 'test.txt')
train_tmp = read_data('train.txt')
train_x =  train_tmp[0]
train_y =  train_tmp[1]

test_tmp = read_data('test.txt')
test_x = test_tmp[0]
test_y = test_tmp[1]

'''
results:
  - k = 1:   0.9194139194139194
  - k = 10:  0.9377289377289377
  - k = 100: 0.8864468864468864

with k=1 and train_x and train_y as test set the result becomes 1.0
the test function calculates how many classifications match the test set,
when test set and training set are the same all of the entries are the same

'''

#eval_cancer_classifier(train_x , train_y, (lambda x: is_cancerous_knn(x, train_x, train_y, dist_function=simple_distance, k=1)))    



from sklearn.model_selection import KFold

def cross_validation(train_x, train_y, untrained_classifier):
  """Uses cross-validation (with 5 folds) to evaluate the given classifier.

  Args:
    train_x: Like above.
    train_y: Like above.
    untrained_classifier: Like above, but also needs training data:
       untrained_classifier should be a function taking 3 arguments (train_x, train_y, x).
       For example:
       untrained_classifier = lambda train_x, train_y, x: is_cancerous_knn(x, train_x,
           train_y, simple_distance, k=5)
  Returns:
    A float, like above (the average error rate evaluated across all folds).
  """

  kf = KFold(n_splits=5)
  kf.get_n_splits(train_x)


  error_rate = 0.0

  for train_index, test_index in kf.split(train_x):

    train_set_X = []
    train_set_Y = []
    test_set_X = []
    test_set_Y = []

    for tr_idx in train_index:
      train_set_X.append(train_x[tr_idx])   
      train_set_Y.append(train_y[tr_idx])

    for tst_idx in test_index:
      test_set_X.append(train_x[tst_idx])
      test_set_Y.append(train_y[tst_idx])

    for i in range(0, len(test_set_X)):
      classification = untrained_classifier(train_set_X, train_set_Y, test_set_X[i])
      if(classification != test_set_Y[i]):
        error_rate += 1.0
      
  return error_rate / len(train_x)


cross_validation(train_x, train_y, lambda train_x, train_y, x: is_cancerous_knn(x, train_x,
          train_y, simple_distance, k=9))


def sampled_range(mini, maxi, num):
  if not num:
    return []
  lmini = math.log(mini)
  lmaxi = math.log(maxi)
  ldelta = (lmaxi - lmini) / (num - 1)
  out = [x for x in set([int(math.exp(lmini + i * ldelta)) for i in range(num)])]
  out.sort()

  return out



def find_best_k(train_x, train_y, untrained_classifier_for_k):
  """Uses cross-validation (10 folds) to find the best K for the given classifier.

  Args:
    train_x: Like above.
    train_y: Like above.
    untrained_classifier_for_k: A function that takes FOUR arguments: train_x, train_y, k
       and x. Example:
       lambda train_x, train_y, k, x: is_cancerous_knn(x, train_x, train_y,
          dist_function, k)

  Returns:
    An integer: the ideal value for K in a K-nearest-neighbor classifier.
  """

  best_k = 1.0
  best_error_rate = 1.0

  for i in sampled_range(1, 200, 10):

    current_error_rate = cross_validation(train_x, train_y, lambda train_x, train_y, x: is_cancerous_knn(x, train_x,
          train_y, simple_distance, k=i))

    if(current_error_rate < best_error_rate):
      best_k = i
      best_error_rate = current_error_rate
      print(best_k, best_error_rate)

  return best_k    

'''
the best error rate is 0.07432432432432433 with k=3
'''
find_best_k(train_x, train_y, lambda train_x, train_y, k, x: is_cancerous_knn(x, train_x, train_y,
          simple_distance, k))    

