'''email_preprocessor.py
Preprocess Enron email dataset into features for use in supervised learning algorithms
Hannah Soria
CS 251/2 Data Analysis Visualization
Spring 2023
'''
import re
import os
import numpy as np


def tokenize_words(text):
    '''Transforms an email into a list of words.

    Parameters:
    -----------
    text: str. Sentence of text.

    Returns:
    -----------
    Python list of str. Words in the sentence `text`.

    This method is pre-filled for you (shouldn't require modification).
    '''
    # Define words as lowercase text with at least one alphabetic letter
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())


def count_words(email_path='data/enron'):
    '''Determine the count of each word in the entire dataset (across all emails)

    Parameters:
    -----------
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_emails: int. Total number of emails in the dataset.

    TODO:
    - Descend into the dataset base directory -> class folders -> individual emails.
    - Read each email file as a string.
    - Use the `tokenize_words` function above to chunk it into a list of words.
    - Update the counts of each word in the dictionary.

    Hints:
    - Check out Python functions in the os and os.path modules for walking the directory structure.
    '''
    files = [] # list for the files
    for (dirpath, dirnames, filenames) in os.walk(email_path): # walk through the file path
        for file in filenames: # for every file i filenames
            files.append(os.path.join(dirpath, file)) # add file to list

    word_frequency = {} # create dictionary for the frequencies
    for f in files: # for every file in the list
        with open(f, 'r', encoding='latin-1') as file: # open the file ( use latin- 1 simplest)
            words = tokenize_words(file.read()) # tokenize
            for w in words: # for every word
                if w in word_frequency: # if the word is in the dictionary
                    word_frequency[w] += 1 # add to the count
                else:
                    word_frequency[w] = 1 # if ot start count

    return word_frequency, len(files)

def find_top_words(word_freq, num_features=200):
    '''Given the dictionary of the words that appear in the dataset and their respective counts,
    compile a list of the top `num_features` words and their respective counts.

    Parameters:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_features: int. Number of top words to select.

    Returns:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    counts: Python list. Counts of the `num_features` words in high-to-low count order.
    '''
    word_frequency_sorted = {k: v for k, v in sorted(word_freq.items(), key=lambda item: item[1], reverse = True)} #create sorted dictionary from parameter dictionary
    top_words = list(word_frequency_sorted.keys())[:num_features] # top words list for num_features
    counts = list(word_frequency_sorted.values())[:num_features] # counts for the num_features

    return top_words, counts


def make_feature_vectors(top_words, num_emails, email_path='data/enron'):
    '''Count the occurance of the top W (`num_features`) words in each individual email, turn into
    a feature vector of counts.

    Parameters:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    num_emails: int. Total number of emails in the dataset.
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    feats. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)

    TODO:
    - Descend into the dataset base directory -> class folders -> individual emails.
    - Read each email file as a string.
    - Use `tokenize_words` to chunk it into a list of words.
    - Count the occurance of each word, ONLY THOSE THAT APPEAR IN `top_words`.

    HINTS:
    - Start with your code in `count_words` and modify as needed.
    '''
    files = [] # list for the files
    for (dirpath, dirnames, filenames) in os.walk(email_path): # walk through email path
        if dirpath == 'data/enron': 
            continue
        for file in filenames:
            files.append(os.path.join(dirpath, file)) #add to the list of files

    feats = np.zeros((num_emails, len(top_words))) # create numpy array of num of emails by the length of top words
    y = np.zeros((num_emails,)) # numpy array for y 
    curr_index = 0 # variable to increase index
    for f in files:
        with open(f, 'r', encoding='latin-1') as file:
            if not os.path.basename(f).startswith('.'):
                word_frequency = {word : 0 for word in top_words} # dictionary top words
                if 'ham' in os.path.dirname(f): # set the y class index
                    y[curr_index] = 0 # 0 is for ham
                elif 'spam' in os.path.dirname(f): # set the y class index
                    y[curr_index] = 1 # 1 is for spam
                words = tokenize_words(file.read()) # tokenize the words
                for w in words: 
                    if w in top_words: # if the word is in the dictionary add to count
                        word_frequency[w] += 1
                    else: # if not in dictionary start count
                        word_frequency[w] = 1
                feats[curr_index] = np.array(list(word_frequency.values())) # add to feature vector
                curr_index += 1 # increase index

    return feats, y


def make_train_test_sets(features, y, test_prop=0.2, shuffle=True):
    '''Divide up the dataset `features` into subsets ("splits") for training and testing. The size
    of each split is determined by `test_prop`.

    Parameters:
    -----------
    features. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)
    test_prop: float. Value between 0 and 1. What proportion of the dataset samples should we use
        for the test set? e.g. 0.2 means 20% of samples are used for the test set, the remaining
        80% are used in training.
    shuffle: boolean. Should we shuffle the data before splitting it into train/test sets?

    Returns:
    -----------
    x_train: ndarray. shape=(num_train_samps, num_features).
        Training dataset
    y_train: ndarray. shape=(num_train_samps,).
        Class values for the training set
    inds_train: ndarray. shape=(num_train_samps,).
        The index of each training set email in the original unshuffled dataset.
        For example: if we have originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].
    x_test: ndarray. shape=(num_test_samps, num_features).
        Test dataset
    y_test:ndarray. shape=(num_test_samps,).
        Class values for the test set
    inds_test: ndarray. shape=(num_test_samps,).
        The index of each test set email in the original unshuffled dataset.
        For example: if we have originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].
    '''
    inds = np.arange(y.size)
    if shuffle:
        features = features.copy()
        y = y.copy()

        inds = np.arange(y.size)
        np.random.shuffle(inds)
        features = features[inds]
        y = y[inds]

    # Your code here:
    split_point = int(test_prop * len(features)) # where is split the data

    # split the data in half, test first half, train second half
    inds_test = inds[:split_point]
    inds_train = inds[split_point:]

    # set the correct side of the data to the variable
    x_test = features[inds_test]
    x_train = features[inds_train]
    y_test = y[inds_test]
    y_train = y[inds_train]

    return x_train, y_train, inds_train, x_test, y_test, inds_test


def retrieve_emails(inds, email_path='data/enron'):
    '''Obtain the text of emails at the indices `inds` in the dataset.

    Parameters:
    -----------
    inds: ndarray of nonnegative ints. shape=(num_inds,).
        The number of ints is user-selected and indices are counted from 0 to num_emails-1
        (counting does NOT reset when switching to emails of another class).
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    Python list of str. len = num_inds = len(inds).
        Strings of entire raw emails at the indices in `inds`
    '''
    list_of_str = [] # list for strings of entire raw emails at the indices on 'inds'
    files = [] # list for the files
    for (dirpath, dirnames, filenames) in os.walk(email_path): # walking through the file path
        if dirpath == 'data/enron':
            continue
        for file in filenames:
            files.append(os.path.join(dirpath, file)) # add the files to the list
    for i in range(len(inds)): # for the length of the inds
        with open(files[inds[i]], 'r', encoding = 'latin-1') as file: # open the files
            list_of_str.append(file.read()) # append to the list
    return list_of_str
