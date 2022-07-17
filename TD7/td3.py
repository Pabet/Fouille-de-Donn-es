# coding=utf-8

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
  words_dict = {}
  l0 = []
  l1 = []
  for line in open(sms_file, 'r').readlines():
    words = line.split()
    l = []
    for w in words[1:]:
      if w not in words_dict:
        words_dict[w] = len(words_dict)
      l.append(words_dict[w])
    if words[0] == 'spam':
      l0.append(l)
    else:
      l1.append(l)
  return words_dict, l0, l1

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
    such that “i in documents[j]”.
    If index #i doesn’t appear in any document, its frequency should be zero.
  """
  freq = [0] * num_words
  for doc in documents:
    for x in set(doc):
      freq[x] += 1
  num_docs = len(documents)
  for i in range(num_words):
    freq[i] = float(freq[i]) / num_docs
  return freq

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
  words, spams, hams = tokenize_and_split(sms_file)
  num_words = len(words)
  freq_spam = compute_frequencies(num_words, spams)
  freq_all = compute_frequencies(num_words, spams + hams)
  return (len(spams) / float(len(spams) + len(hams)),
          words,
          [freq_spam[i] / float(freq_all[i]) for i in range(num_words)])

def naive_bayes_train_ham(sms_file):
  """Like naive_bayes_train, but for 'ham' messages (not spam)."""
  words, spams, hams = tokenize_and_split(sms_file)
  num_words = len(words)
  freq_ham = compute_frequencies(num_words, hams)
  freq_all = compute_frequencies(num_words, spams + hams)
  return (len(hams) / float(len(spams) + len(hams)),
          words,
          [freq_ham[i] / float(freq_all[i]) for i in range(num_words)])

def naive_bayes_predict(spam_ratio, words, spamicity, sms):
  """Performs the "prediction" phase of the Naive Bayes estimator.

  You should use the simple formula:
  P(spam|words in sms) = spam_ratio * Product[word in sms]( P(word|spam) / P(word) )

  Args:
    spam_ratio: see output of naive_bayes_train
    words: see output of naive_bayes_train
    spamicity: see output of naive_bayes_train
    sms: a string (which you can tokenize to obtain a list of words)

  Returns:
    The estimated probability that the given sms is a spam.
  """
  p = spam_ratio
  for w in set(sms.split()):
    if w not in words:
      continue
    p *= spamicity[words[w]]
  return p

def naive_bayes_eval(test_sms_file, f, log=False):
  """Evaluates a spam classifier.

  Args:
    test_sms_file: a string, the name of the input 'test' SMS data file.
    f: a function. f(sms), where sms is a string (like "Hi. Where are you?",
        should return 1 if sms is classified as spam, and 0 otherwise.

  Returns:
    A pair of floats (recall, precision): 'recall' is the ratio (in [0,1]) of
    spams in the test file that were successfully identified as spam, and
    'Precision' is the ratio of predicted spams that were actually spam.
  """
  num_predicted_spams = 0
  num_actual_spams = 0
  num_actual_and_predicted_spams = 0
  for line in open(test_sms_file, 'r').readlines():
    is_spam = (line[:4] == 'spam')
    sms = line[5:] if is_spam else line[4:]
    if f(sms) == 1:
      num_predicted_spams += 1
      if is_spam:
        num_actual_and_predicted_spams += 1
      elif log:
        print('False positive: ' + sms)
    elif is_spam and log:
      print('Undetected spam: ' + sms)
    if is_spam:
      num_actual_spams += 1
  return (num_actual_and_predicted_spams / float(num_actual_spams) if num_actual_spams else 1.0,
          num_actual_and_predicted_spams / float(num_predicted_spams) if num_predicted_spams else 1.0)

spamicity = []
hamicity = []
spam_ratio = None
ham_ratio = None
words_for_spam = {}
words_for_ham = {}

def run_experiment(min_freq_spam = 0.01, min_freq_ratio_spam_ham = 0.5,
                   min_freq_ham = 0.01, min_freq_ratio_ham_spam = 0.5,
                   dry_run=True):
  global spamicity, hamicity, spam_ratio, ham_ratio, words_for_spam, words_for_ham
  split_lines('SMSSpamCollection', 0, 'train', 'test')
  spam_ratio, words, spamicity = naive_bayes_train('train')
  ham_ratio, words, hamicity = naive_bayes_train_ham('train')

  words, spams, hams = tokenize_and_split('train')
  num_words = len(words)
  words_for_spam = {}
  words_for_ham = {}
  freq_spam = compute_frequencies(num_words, spams)
  freq_ham = compute_frequencies(num_words, hams)
  for w, i in words.items():
    if freq_spam[i] > min_freq_spam and freq_spam[i] > min_freq_ratio_spam_ham*freq_ham[i]:
      words_for_spam[w] = i
    if freq_ham[i] > min_freq_ham and freq_ham[i] > min_freq_ratio_ham_spam*freq_spam[i]:
      words_for_ham[w] = i
  if dry_run:
    return
  # Write data files to visualize the classification.
  suffix = '_%f_%f_%f_%f' % (min_freq_spam, min_freq_ratio_spam_ham,
                             min_freq_ham, min_freq_ratio_ham_spam)
  with open('spams' + suffix, 'w') as spams, open('hams' + suffix, 'w') as hams:
    for line in open('test','r').readlines():
      is_spam = (line[:4]=='spam')
      sms = line[5:-1] if is_spam else line[4:-1]
      y_spamicity = naive_bayes_predict(spam_ratio, words_for_spam, spamicity, sms)
      x_hamicity = naive_bayes_predict(ham_ratio, words_for_ham, hamicity, sms)
      out = '%f %f "%s"\n' % (x_hamicity, y_spamicity, sms)
      (spams if is_spam else hams).write(out)
  print(('Wrote files for min_freq_spam=%f, min_freq_ratio_spam_ham=%f,' +
         ' min_freq_ham=%f, min_freq_ratio_ham_spam=%f.') % (
           min_freq_spam, min_freq_ratio_spam_ham,
           min_freq_ham, min_freq_ratio_ham_spam))
  print(("To plot, run this command:\ngnuplot -e \"set logscale x;" +
         " set logscale y; set out 'sms%s.png'; set term png size 1024,1024;" +
         " set xlabel 'p_{ham}'; set ylabel 'p_{spam}';" +
         " plot 'hams%s','spams%s' with points\"") % (suffix, suffix, suffix))

run_experiment()

def classify_spam(sms):
  """Returns True is the message 'sms' is predicted to be a spam."""
  # Measurement from graph 'sms_0.010000_0.500000_0.001000_0.500000.png':
  # A good dividing line seems to be at y=26*x**7.
  x = naive_bayes_predict(ham_ratio, words_for_ham, hamicity, sms)
  y = naive_bayes_predict(spam_ratio, words_for_spam, spamicity, sms)
  return y > 26*x**7

