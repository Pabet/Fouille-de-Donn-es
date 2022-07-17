#%%
import random

def split_lines(input, seed, output1, output2):
    """
    Distributes the lines of 'input' to 'output1' and 'output2' pseudo-randomly.

    The output files should be approximately balanced (50/50 chance for each line
    to go either to output1 or output2).
    
    Args:
        input: a string, the name of the input file.
        seed: an integer, the seed of the pseudo-random generator used. The split
        should be different with different seeds. Conversely, using the same
        seed and the same input should yield exactly the same outputs.
        output1: a string, the name of the first output file.
        output2: a string, the name of the second output file.
    """
    
    random.seed(seed)   
    first_file = open(output1, 'w')
    second_file = open(output2, 'w')

    for line in open(input, 'r').readlines():
        if random.randint(0, 1) == 0:
            first_file.write(line)
        else:
            second_file.write(line)

   
    return first_file.close() and second_file.close()
   


#split_lines('testset', 100, 'output1', 'output2')    

#%%

def tokenize_and_split(sms_file):
    """Parses and tokenizes the sms data, splitting 'spam' and 'ham' messages.
  
  Args:
    sms_file: a string, the name of the input SMS data file.

  Returns:
    A triple (words, l0, l1):
    - words is a dictionary mapping each word to a unique, dense 'word index'.
      The word indices must be in [0...len(words)-1].
    - l0 is a list of the 'spam' messages, encoded as lists of word indices.
    - l1 is like l0, but for 'ham' messages.
    """

    words = dict()
    ham_list = []
    spam_list = []
    index_counter = 0

    for line in open(sms_file, 'r').readlines():
        encoded_message = []
        line_list = line.split()
            
        for word in line_list:
        
            word_index = index_counter
            if not word == 'ham' and not word == 'spam':
                if not word in words: 
                    words[word] = index_counter
                    index_counter += 1
                else:
                    word_index = words[word]
                encoded_message.append(word_index)
            
        if line_list[0] == 'ham':     
            ham_list.append(encoded_message)
        else:
            spam_list.append(encoded_message)  

    #print(words)
    return (words, spam_list, ham_list)


#tokenize_and_split('output1')    


#%%

def compute_frequencies(num_words, documents):
    """Computes the frequency of words in a corpus of documents.
    
  Args:
    num_words: the number of words that exist. Words will be integers in
        [0..num_words-1].
    documents: a list of lists of integers. Like the l0 or l1 output of
        tokenize_and_split().

  Returns:
    A list of floats of length num_words: element #i will be the ratio
    (in [0..1]) of documents containing i, i.e. the ratio of indices j
    such that "i in documents[j]".
    If index #i doesn't appear in any document, its frequency should be zero.
    """
    results = []

    for word in range(num_words):
        occurences = 0
        for integers in documents:
            if word in integers:
                occurences += 1
        results.append(occurences/len(documents))

    return results



def naive_bayes_train(sms_file):
    """Performs the "training" phase of the Naive Bayes estimator.
    
  Args:
    sms_file: a string, the name of the input SMS data file.

  Returns:
    A triple (spam_ratio, words, spamicity) where:
    - spam_ratio is a float in [0..1] and is the ratio of SMS marked as 'spam'.
    - words is the dictionary output by tokenize_and_split().
    - spamicity is a list of num_words floats, where num_words = len(words) and
      spamicity[i] = (ratio of spams containing word #i) /
                     (ratio of SMS (spams and hams) containing word #i)
    """

    words, spam_list, ham_list = tokenize_and_split(sms_file)

    spam_ratio = len(spam_list)/(len(ham_list)+len(spam_list))

    spam_frequencies = compute_frequencies(len(words), spam_list)
    ham_frequencies = compute_frequencies(len(words), ham_list)

    spamicity = []

    for i in range(len(words)):
        spamicity.append(spam_frequencies[i]/(spam_frequencies[i]*spam_ratio+ham_frequencies[i]*(1-spam_ratio)))

    return (spam_ratio, words, spamicity)

#naive_bayes_train('test3_0')

#%%

def naive_bayes_predict(spam_ratio, words, spamicity, sms):
    """Performs the "prediction" phase of the Naive Bayes estimator.

  You should use the simple formula:
  P(spam|words in sms) = spam_ratio * Product[word in sms]( P(word|spam) / P(word) )
  Make sure you skip (i.e. ignore) the SMS words that are unknown (not in 'words').
  BE CAREFUL: if a word is repeated in the sms, it shouldn't appear twice here!
    
  Args:
    spam_ratio: see output of naive_bayes_train
    words: see output of naive_bayes_train
    spamicity: see output of naive_bayes_train
    sms: a string (which you can tokenize to obtain a list of words)

  Returns:
    The estimated probability that the given sms is a spam.
    """

    sms_list = sms.split()
    used_indices = []


    word_probabilities = []
    for word in sms_list:
        index = 0
        if word in words:
            index = words[word]
            if not index in used_indices:
                word_probabilities.append(spamicity[index])
                used_indices.append(index)

    probability_product = spam_ratio
    for i in word_probabilities:
        probability_product *= i

    return probability_product    

#print(naive_bayes_predict(0.3333333333333, {'Hello': 0, 'World': 1, 'awesome': 2, 'stuff': 3, '?': 4}, [1.0, 0.0, 3.0, 3.0, 0.0], 'Hello dude'))

#%%

def naive_bayes_eval(test_sms_file, f):
    """Evaluates a spam classifier.
  
  Args:
    test_sms_file: a string, the name of the input 'test' SMS data file.
    f: a function. f(sms), where sms is a string (like "Hi. Where are you?",
        should return 1 if sms is classified as spam, and 0 otherwise.
  
  Returns:
    A pair of floats (recall, precision): 'recall' is the ratio (in [0,1]) of
    spams in the test file that were successfully identified as spam, and
    'Precision' is the ratio, among all sms that were predicted as spam by f, of
    sms that were indeed spam.
    """

    identified_as_spam = 0.0
    spam_in_file = 0.0
    identified_as_spam_real_spam = 0.0
    recall_counter = 0.0

    for line in open(test_sms_file, 'r').readlines(): 
        spam_index = f(line)
        line_list = line.split()

        if line_list[0] == 'spam':
            spam_in_file += 1
            if spam_index == 1:
                recall_counter += 1

        if spam_index == 1:
            identified_as_spam += 1
            if line_list[0] == 'spam':
                identified_as_spam_real_spam += 1

    if identified_as_spam == 0:
        return (0,1)   
    if spam_in_file == 0:
        return (1,1)

    return (recall_counter/spam_in_file, identified_as_spam_real_spam/identified_as_spam)

spam_ratio, words, spamicity = naive_bayes_train('test3_0')
print(naive_bayes_eval('test3_0',
    lambda x:naive_bayes_predict(spam_ratio, words, spamicity, x)>0.5))
   