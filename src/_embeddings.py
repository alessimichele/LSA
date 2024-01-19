"""
Code for Information Retrieval course project @ University of Trieste, MSc in Data Science & Scientific Computing AA 2023/2024.
Author: Michele Alessi

This file contains the code to build the inverted index and compute the TF-IDF matrix and WC matrix given a corpus.

*NOTE*: during actual computation within the IR class, the corpus is processed using scikit-learn CountVectorizer (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) and TfidfVectorizer (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).
This is done for efficiency reasons, as the scikit-learn implementation is much faster than the one implemented here.
Furthermore, the scikit-learn implementation allows easy and efficient integration with preprocessing steps such as stemming.
This is crucial for the training of the neural networks, as decrease significantly the number of neurons in the input layer, thus reducing the computational cost of the training.
"""

from collections import defaultdict
from math import log
import numpy as np

def make_inv_index(corpus, sort = False, SW = 0.95):
    """
    Function computing the inverted index for the corpus.
    For each document in the corpus, the function iterates over each term in the document and adds the document id to the set of documents containing the term.

    Note: defaultdict is used to create a dictionary: if the key is not present, it is initialized as a new key with an empty set as its value.

    Args:
        corpus (list): List of documents.
        sort (bool, optional): If True, the inverted index is sorted alphabetically. Defaults to False.
        SW (float, optional): If not None, the terms appearing in more than SW*N documents are removed from the index. Defaults to 0.95.

    Returns:
        dict: Inverted index.
    """
    N = len(corpus)
    inv_index = defaultdict(set)
    for docid, article in enumerate(corpus):
        for term in article:
            inv_index[term].add(docid)

    if SW != None:
        for term in list(inv_index.keys()):
            if len(inv_index[term]) >= SW*N:
                # Remove the term from the index if it appears in more than 95% of the documents.
                del inv_index[term]

    if sort:
        inv_index = dict(sorted(inv_index.items()))
    
    return inv_index


def DF(term, inv_index):
    """
    Funciton computing Document Frequency for a term. It is the number of documents containing the term, i.e. the length of the posting list for the term.

    Args:
        term (str): Term.
        inv_index (dict): Inverted index.

    Returns:
        int: Document Frequency.
    """
    return len(inv_index[term])


def IDF(term, inv_index, corpus):
    """
    Function computing Inverse Document Frequency for a term. 
    It is the logarithm of the ratio of the total number of documents in the corpus to the document frequency for the term.

    Args:
        term (str): Term.
        inv_index (dict): Inverted index.
        corpus (list): List of documents.

    Returns:
        float: Inverse Document Frequency.
    """
    return log(len(corpus)/DF(term, inv_index))

def TF(term, docid, corpus):
    """
    Function computing Term Frequency for a term in a document. 
    It is the number of times the term appears in the document.
    """
    return corpus[docid].count(term)

def TF_IDF(term, docid, inv_index, corpus):
    """
    Function computing TF-IDF for a term in a document.

    Args:
        term (str): Term.
        docid (int): Document id.
        inv_index (dict): Inverted index.
        corpus (list): List of documents.

    Returns:
        float: TF-IDF.
    """
    return TF(term, docid, corpus) * IDF(term, inv_index, corpus)


def TF_IDF_matrix(corpus, inv_index):
    """
    Function computing TF-IDF matrix for the corpus.

    Args:
        corpus (list): List of documents.
        inv_index (dict): Inverted index.

    Returns:
        np.array: TF-IDF matrix.
    """
    # Initialize an empty matrix.
    matrix = np.zeros((len(corpus), len(inv_index)))

    # Iterate over each document in the corpus.
    for docid, article in enumerate(corpus):
        # Iterate over each term in the document.
        for termid, term in enumerate(inv_index):
            # Compute TF-IDF for the term in the document.
            matrix[docid, termid] = TF_IDF(term, docid, inv_index, corpus)
    # Normalize the matrix.
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    # Return the matrix.
    return matrix

def WC_matrix(corpus, inv_index):
    """
    Function computing Word Count matrix for the corpus.

    Args:
        corpus (list): List of documents.
        inv_index (dict): Inverted index.

    Returns:
        np.array: Word Count matrix.
    """
    # Initialize an empty matrix.
    matrix = np.zeros((len(corpus), len(inv_index)))

    # Iterate over each document in the corpus.
    for docid, article in enumerate(corpus):
        # Iterate over each term in the document.
        for termid, term in enumerate(inv_index):
            # Compute TF-IDF for the term in the document.
            matrix[docid, termid] = TF(term, docid, corpus)

    # Return the matrix.
    return matrix