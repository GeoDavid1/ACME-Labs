"""Volume 3: Naive Bayes Classifiers."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import math


class NaiveBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages into spam or ham.
    '''
    # Problem 1
    def fit(self, X, y):
        '''
        Compute the values P(C=Ham), P(C=Spam), and P(x_i|C) to fit the model.

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''

        # Find total number of samples and total spam and total ham samples
        num_samples = len(X)
        spam_num_samples = len(y[y == 'spam'])
        ham_num_samples = len(y[y == 'ham'])
        
        # Find which samples are spam and which ones are ham
        spam_indices = y[y == 'spam'].index
        ham_indices = y[y== 'ham'].index
        
        # Split the samples into separate words
        spam_mail = X[spam_indices].str.split()
        ham_mail = X[ham_indices].str.split()
        
        # Get total number of words in spam and ham mail
        spam_total_count = sum(spam_mail.apply(len))
        ham_total_count = sum(ham_mail.apply(len))
        
        total_list_words = []
        
        # Get occurrences of each word in class spam mail
        for spam_message in spam_mail:
            for word in spam_message:
                total_list_words.append(word)
                
                
        # Get occurrences of each word in class spam mail
        for ham_message in ham_mail:
            for word in ham_message:
                total_list_words.append(word)
                
        total_list_words = list(set(total_list_words))
        
         # Initialize probability dictionaries
        spam_dict = {word: 0 for word in total_list_words}
        ham_dict = {word: 0 for word in total_list_words}
        
        # Get occurrences of each word in class ham mail           
        for spam_message in spam_mail:
            for word in spam_message:
                spam_dict[word] += 1
                     
        # Get occurrences of each word in class ham mail           
        for ham_message in ham_mail:
            for word in ham_message:
                ham_dict[word] += 1
        
        # Make dictionaries of spam and ham probabilities (Use Gaussian smoothing)       
        self.spam_probs = {key: (value + 1) / (spam_total_count + 2) for key, value in spam_dict.items()}
        self.ham_probs = {key: (value  + 1) / (ham_total_count + 2) for key, value in ham_dict.items()}
        self.spam_prop = spam_num_samples / num_samples
        self.ham_prop = ham_num_samples / num_samples
        
        return self
                    
        
    # Problem 2
    def predict_proba(self, X):
        '''
        Find ln(P(C=k,x)) for each x in X and for each class.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Log probability each message is ham or spam.
                Column 0 is ham, column 1 is spam.
        '''
        # Initialize log probabilities array
        log_probabilities = []
    
        # Add log probabilities of spam proportion and ham proportion
        for message in X:
            log_prob_spam = np.log(self.spam_prop)
            log_prob_ham = np.log(self.ham_prop)
        
            # Split the message into words
            words = message.split() 
        
            # Iterate through the words of the message
            for word in words:
                # Add log probability of word for spam
                if word in self.spam_probs:
                    log_prob_spam += np.log(self.spam_probs[word])
                else:
                    log_prob_spam += np.log(0.5) 
                
                # Add log probability of word for ham
                if word in self.ham_probs:
                    log_prob_ham += np.log(self.ham_probs[word])
                else:
                    log_prob_ham += np.log(0.5)  
            
            # Append probabilities
            log_probabilities.append([log_prob_ham, log_prob_spam])
    
        # Return log probabilities
        return np.array(log_probabilities)
        
            
    # Problem 3
    def predict(self, X):
        '''
        Predict the labels of each row in X, using self.predict_proba().
        The label will be a string that is either 'spam' or 'ham'.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        
        # Initialize list of predictions
        predictions = []
        
        # Get log probabilities
        probabilities = self.predict_proba(X)
        
        # Classify each sample as 'ham' or 'spam'
        for log_prob in probabilities:
            if log_prob[0] >= log_prob[1]:
                predictions.append('ham')
                
            else:
                predictions.append('spam')
        
        # Return predictions    
        return np.array(predictions)
        

def prob4():
    """
    Create a train-test split and use it to train a NaiveBayesFilter.
    Predict the labels of the test set.
    
    Compute and return the following two values as a tuple:
     - What proportion of the spam messages in the test set were correctly identified by the classifier?
     - What proportion of the ham messages were incorrectly identified?
    """
    
    # Read in data; get messages and labels
    df = pd.read_csv('sms_spam_collection.csv')
    X = df.Message
    y = df.Label
    
    # Perform a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    
    # Create a NaiveBayesFilter
    nb_prob4 = NaiveBayesFilter()
    
    # Fit the data
    nb_prob4.fit(X_train, y_train)
    
    # Get predicted labels for y_test
    predictions = nb_prob4.predict(X_test)
    
    # Sort which ones are spam and ham
    y_test = y_test.reset_index(drop = True)
    y_test_spam_indices = y_test[y_test == 'spam'].index
    y_test_ham_indices = y_test[y_test == 'ham'].index

    # Find proportion of spam messages that are correct and ham messages that are incorrect
    spam_matches = np.sum(predictions[y_test_spam_indices] == y_test[y_test_spam_indices])
    ham_matches = np.sum(predictions[y_test_ham_indices] == y_test[y_test_ham_indices])
    spam_proportion = spam_matches / len(y_test_spam_indices)
    ham_proportion = 1 - ham_matches / len(y_test_ham_indices)
    
    # Return proportions
    return (spam_proportion, ham_proportion)


    

# Problem 5
class PoissonBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    This classifier assumes that words are distributed like
    Poisson random variables.
    '''
    def fit(self, X, y):
        '''
        Compute the values P(C=Ham), P(C=Spam), and r_{i,k} to fit the model.

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''
        
        # Find total number of samples and total spam and total ham samples
        num_samples = len(X)
        spam_num_samples = len(y[y == 'spam'])
        ham_num_samples = len(y[y == 'ham'])
        
        # Find which samples are spam and which ones are ham
        spam_indices = y[y == 'spam'].index
        ham_indices = y[y== 'ham'].index
        
        # Split the samples into separate words
        spam_mail = X[spam_indices].str.split()
        ham_mail = X[ham_indices].str.split()
        
        # Get total number of words in spam and ham mail
        spam_total_count = sum(spam_mail.apply(len))
        ham_total_count = sum(ham_mail.apply(len))
        
        total_list_words = []
        
        # Get occurrences of each word in class spam mail
        for spam_message in spam_mail:
            for word in spam_message:
                total_list_words.append(word)
                
                
        # Get occurrences of each word in class spam mail
        for ham_message in ham_mail:
            for word in ham_message:
                total_list_words.append(word)
                
        total_list_words = list(set(total_list_words))
        
        
        # Initialize probability dictionaries
        spam_dict = {word: 0 for word in total_list_words}
        ham_dict = {word: 0 for word in total_list_words}
        
        # Get occurrences of each word in class spam mail
        spam_wordcount = 0
        for spam_message in spam_mail:
            spam_wordcount += len(spam_message)
            for word in spam_message:
                spam_dict[word] += 1
                    
        # Get occurrences of each word in class ham mail  
        ham_wordcount = 0         
        for ham_message in ham_mail:
            ham_wordcount += len(ham_message)
            for word in ham_message:
                    ham_dict[word] += 1
        
        # Make dictionaries of spam and ham probabilities         
        self.spam_rates = {key: (value + 1) / (spam_total_count + 2) for key, value in spam_dict.items()}
        self.ham_rates = {key: (value  + 1) / (ham_total_count + 2) for key, value in ham_dict.items()}
        self.spam_prop = spam_num_samples / num_samples
        self.ham_prop = ham_num_samples / num_samples
        self.spam_wordcount = spam_wordcount
        self.ham_wordcount = ham_wordcount
    

    def predict_proba(self, X):
        '''
        Find ln(P(C=k,x)) for each x in X and for each class.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Log probability each message is ham or spam.
                Column 0 is ham, column 1 is spam.
        '''
        
        # Initialize log probabilities
        log_probabilities = []

        # Iterate through messages
        for message in X:
            
            # Add in log prob of spam and ham
            log_prob_spam = np.log(self.spam_prop)
            log_prob_ham = np.log(self.ham_prop)
            
            # Find length, words, and counts of words in message
            n = len(message.split())
            words, counts = np.unique(message.split(), return_counts = True)
            
            # Iterate through each word
            for word, count in zip(words,counts):
                
                # Get poisson log-pmf if word in dictionary (spam)
                if word in self.spam_rates:
                    rate = self.spam_rates[word]
                    lambda_parameter = rate * n
                    w = count
                    log_prob_spam += stats.poisson.logpmf(w, lambda_parameter)
                  
                # Get poisson log-pmf if word not in dictionary (spam)
                else:
                    rate = 1/(self.spam_wordcount + 2)
                    lambda_parameter = rate * n
                    w = count
                    log_prob_spam += stats.poisson.logpmf(w, lambda_parameter)
                
                # Get poisson log-pmf if word in dictionary (ham) 
                if word in self.ham_rates:
                    rate = self.ham_rates[word]
                    lambda_parameter = rate * n
                    w = count
                    log_prob_ham += stats.poisson.logpmf(w, lambda_parameter)
                
                # Get poisson log-pmf if word not in dictionary (ham)   
                else:
                    rate = 1/(self.ham_wordcount + 2)
                    lambda_parameter = rate * n
                    w = count
                    log_prob_ham += stats.poisson.logpmf(w, lambda_parameter)
            
            # Append log probabilities  
            log_probabilities.append([log_prob_ham, log_prob_spam])
        
        # Return log probabilities
        return np.array(log_probabilities)
                    
            
    def predict(self, X):
        '''
        Predict the labels of each row in X, using self.predict_proba().
        The label will be a string that is either 'spam' or 'ham'.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        
        # Initialize list of predictions
        predictions = []
        
        # Get log probabilities
        probabilities = self.predict_proba(X)
        
        # Classify each sample as 'ham' or 'spam'
        for log_prob in probabilities:
            if log_prob[0] >= log_prob[1]:
                predictions.append('ham')
            
            else:
                predictions.append('spam')

        # Return predictions
        return np.array(predictions)
    

def prob6():
    """
    Create a train-test split and use it to train a PoissonBayesFilter.
    Predict the labels of the test set.
    
    Compute and return the following two values as a tuple:
     - What proportion of the spam messages in the test set were correctly identified by the classifier?
     - What proportion of the ham messages were incorrectly identified?
    """
    
    # Read in data; get messages and labels
    df = pd.read_csv('sms_spam_collection.csv')
    X = df.Message
    y = df.Label
    
    # Perform a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    
    # Create a NaiveBayesFilter
    pb_prob6 = PoissonBayesFilter()
    
    # Fit the data
    pb_prob6.fit(X_train, y_train)
    
    # Get predicted labels for y_test
    predictions = pb_prob6.predict(X_test)
    
    # Sort which ones are spam and ham
    y_test = y_test.reset_index(drop = True)
    y_test_spam_indices = y_test[y_test == 'spam'].index
    y_test_ham_indices = y_test[y_test == 'ham'].index
    
    # Find proportion of spam messages that are correct and ham messages that are incorrect
    spam_matches = np.sum(predictions[y_test_spam_indices] == y_test[y_test_spam_indices])
    ham_matches = np.sum(predictions[y_test_ham_indices] == y_test[y_test_ham_indices])
    spam_proportion = spam_matches / len(y_test_spam_indices)
    ham_proportion = 1 - ham_matches / len(y_test_ham_indices)
    
    # Return proportions
    return (spam_proportion, ham_proportion) 


# Problem 7
def sklearn_naive_bayes(X_train, y_train, X_test):
    '''
    Use sklearn's methods to transform X_train and X_test, create a
    naïve Bayes filter, and classify the provided test set.

    Parameters:
        X_train (pandas.Series): messages to train on
        y_train (pandas.Series): labels for X_train
        X_test  (pandas.Series): messages to classify

    Returns:
        (ndarray): classification of X_test
    '''
    
    # Transform train data
    vectorizer = CountVectorizer()
    train_counts = vectorizer.fit_transform(X_train)
    
    # Fit Naive Bayes model to training data
    clf = MultinomialNB()
    clf = clf.fit(train_counts, y_train)
    
    # Classify test data and return classification of X_test
    test_counts = vectorizer.transform(X_test)
    labels = clf.predict(test_counts)  
    return labels
