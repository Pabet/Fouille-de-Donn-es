#%%

def read_dataset(filename):
    """ 
    Reads the file at the given path that should contain one
    type and one text separated by a tab on each line, and returns
    pairs of type/text.
    
    Args:
      filename: a file path.
    Returns:
      a list of (type, text) tuples. Each type is either 0 or 1.
    """
    type_text_list = []
    
    for line in open(filename, 'r').readlines():
        classifier = line.split()[0]
        cleaned_line = line.split(None, 1)[1]
        if(classifier == 'spam'):
            type_text_list.append((1, cleaned_line))
        else:
            type_text_list.append((0, cleaned_line))
    return type_text_list


#print(read_dataset('SMSSpamCollection'))        


def spam_count(pairs):
    """ 
    Returns the number of spams from a list of (type, text) tuples.
    Args:
      pairs: a list of (type, text) tuples.
    Returns:
      an integer representing the number of spams.
    """
    spam_counter = 0
    for tp in pairs:
        if tp[0] == 1:
            spam_counter += 1
    return spam_counter      


#print(spam_count(read_dataset('SMSSpamCollection'))) 
from sklearn.feature_extraction.text import TfidfVectorizer

def transform_text(pairs):
    """
    Transforms the pair data into a matrix X containing TF-IDF values
    for the messages and a vector y containing 0s and 1s (for hams and 
    spams respectively). 
    Row i in X corresponds to the i-th element of y.

    Args:
      pairs: a list of (type, message) tuples.
    Returns:
      A pair (X, Y) with:
      X: a sparse TF-IDF matrix where each row represents a message and 
         each column represents a word.
      Y: a vector whose i-th element is 0 if the i-th message is a ham, 
         else 1.
    """
    text = []
    Y = []

    for pair in pairs:
        text.append(pair[1])
        Y.append(pair[0])

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text)  
    return (X,Y)
"""
X,Y = transform_text(read_dataset('test'))    
for x,y in zip(*X.nonzero()):
  print("[%d][%d] = %s" % (x, y, X[x, y]))
"""


from sklearn.cluster import KMeans

def kmeans_and_most_common_words(pairs, K, P):
    """
    Applies TF-IDF and then the K-Means algorithm with the given K on the
    pair data. Then returns the most common P words in each cluster.

    Args:
      pairs: a list of (type, message) tuples.
    Returns:
      A list of K lists, each containing P strings: the “most
      characteristic” words of each cluster. By “most characteristic, we
      mean the highest TF-IDF score, averaged over the entire cluster.
    """

    (X,Y) = transform_text(pairs)
    kmeans = KMeans(n_clusters=K).fit(X)
    y = kmeans.labels_
    
    for i in range()
     



kmeans_and_most_common_words(read_dataset('test'), 2, 5)