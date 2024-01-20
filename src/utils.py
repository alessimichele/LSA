"""
Code for Information Retrieval course project @ University of Trieste, MSc in Data Science & Scientific Computing A.Y. 2023/2024.
Author: Michele Alessi

This file contains some useful functions for the project.
"""

import re
import string
from torch.utils.data import Dataset
from nltk.stem.snowball import SnowballStemmer


def compute_pre_rec(result, relevances):
    """
    This function computes the precision and recall for each query in a .QRY file, given the results retrieved and the true relevant documents.
    It is used to assess directly on the .QRY file the performance of the model.

    Args:
        result: (list) List of dictionaries. result[i] is the dictionary of documents retrieved for query in queries[i] (result[i] = {doc_id: score}})
        relevances: (list) List of lists. relevances[i] is the list of relevant documents for query in queries[i].

    Returns:
        precision: (list) List of floats. precision[i] is the precision for query in queries[i].
        recall: (list) List of floats. recall[i] is the recall for query in queries[i].
    """
    precision = []
    recall = []
    for i in range(len(result)):
        if result[i] is not None and relevances[i] is not None:
            rec = (len(set(result[i].keys()).intersection(set(relevances[i])))/len(set(relevances[i]))) 
            prec = (len(set(result[i].keys()).intersection(set(relevances[i])))/len(set(result[i].keys())))
        precision.append(prec)
        recall.append(rec)

    return precision, recall


"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""


def custom_preprocessor(text):
    """
    This function is used as custom preprocessor for the CountVectorizer and TfidfVectorizer.

    Args:
        text: (str) Text to preprocess.

    Returns:
        text: (str) Preprocessed text.
    """

    # remove emails
    text = re.sub(r'\S*@\S*\s?', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # remove words with repeated characters
    text = re.sub(r'(\w)\1+', r'\1', text)

    # remove 3 len words
    text = re.sub(r'\b\w{1,3}\b', '', text)

    # remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # remove word longer than 15 characters
    text = re.sub(r'\b\w{15,}\b', '', text)

    # Convert to lowercase
    text = text.lower()

    # stemmer
    stemmer = SnowballStemmer("english")
    text = ' '.join([stemmer.stem(word) for word in text.split()])

    
    return text


"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

class TextDataset(Dataset):
    """
    Class to create a labelled dataloader for the 20newsgroups dataset.
    Each sample is a tuple (article, label).

    Args:
        X: (list) List of articles.
        y: (list) List of labels.

    Returns:
        X[idx]: (str) Article.
        y[idx]: (int) Label.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class TextDatasetUnlabeled(Dataset):
    """
    Class to create an unlabelled dataloader for the 20newsgroups dataset.
    Each sample is an article.

    Args:
        X: (list) List of articles.

    Returns:
        X[idx]: (str) Article.
    """
    
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]
    


