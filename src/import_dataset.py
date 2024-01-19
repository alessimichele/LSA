"""
Code for Information Retrieval course project @ University of Trieste, MSc in Data Science AA 2023/2024.
Author: Michele Alessi

This file contains the code to import the datasets inside the ../data/ folder.
"""

import re

def import_TIME():
    """
    This function imports the content of ../data/time/ folder.

    num of articles: 423
    num of queries: 83

    Returns:
        articles: (list) List of strings. articles[i-1] is the i-th article of TIME.ALL file.
        splitted_articles: (list) List of lists. splitted_articles[i-1] is the i-th article of TIME.ALL file, splitted in words.
        queries: (list) List of strings. queries[i-1] is the i-th query of TIME.QUE file.
        splitted_queries: (list) List of lists. splitted_queries[i-1] is the i-th query of TIME.QUE file, splitted in words.
        relevances: (list) List of lists. relevances[i] is the list of relevant documents for query in queries[i].
    """
    splitted_articles = []
    with open('./data/time/TIME.ALL', 'r') as f:
        tmp = []
        for row in f:
            if row.startswith("*TEXT"):
                if tmp != []:
                    splitted_articles.append(tmp)
                tmp = []
            else:
                row = re.sub(r'[^a-zA-Z\s]+', '', row)
                tmp += row.split()
        if tmp != []:
            splitted_articles.append(tmp)

    articles = []
    for doc in splitted_articles:
        articles.append(' '.join(doc))

    splitted_queries = []
    with open('./data/time/TIME.QUE', 'r') as f:
        tmp = []
        for row in f:
            if row.startswith("*FIND"):
                if tmp != []:
                    splitted_queries.append(tmp)
                tmp = []
            else:
                row = re.sub(r'[^a-zA-Z\s]+', '', row)
                tmp += row.split()
        if tmp != []:
            splitted_queries.append(tmp)

    queries = []
    for query in splitted_queries:
        queries.append(' '.join(query))


    relevances = []
    with open('./data/time/TIME.REL', 'r') as f:
        # iterate over the rows in the file
        for row in f:
            # if the row is empty, skip it
            if not row.strip():
                continue
            # read the content of the row as integers and split it
            content = [int(x) for x in row.split()]
            relevances.append(content[1:])

    return articles, splitted_articles, queries, splitted_queries, relevances

def import_MED():
    """
    This function imports the content of ../data/med/ folder.

    num of articles: 1033
    num of queries: 29

    Returns:
        articles: (list) List of strings. articles[i-1] is the i-th article of MED.ALL file.
        splitted_articles: (list) List of lists. splitted_articles[i-1] is the i-th article of MED.ALL file, splitted in words.
        queries: (list) List of strings. queries[i-1] is the i-th query of MED.QUE file.
        splitted_queries: (list) List of lists. splitted_queries[i-1] is the i-th query of MED.QUE file, splitted in words.
        relevances: (list) List of lists. relevances[i] is the list of relevant documents for query in queries[i].
    """
    splitted_articles = []
    with open('./data/med/MED.ALL', 'r') as f:
        tmp = []
        for row in f:
            if row.startswith(".I"):
                while not row.startswith(".W"):
                    row = f.readline()
                if tmp != []:
                    splitted_articles.append(tmp)
                tmp = []
            else:
                row = re.sub(r'[^a-zA-Z\s]+', '', row)
                row = row.upper()
                tmp += row.split()
        if tmp != []:
            splitted_articles.append(tmp)

    articles = []
    for doc in splitted_articles:
        articles.append(' '.join(doc))

    splitted_queries = []
    with open('./data/med/MED.QRY', 'r') as f:
        tmp = []
        for row in f:
            if row.startswith(".I"):
                while not row.startswith(".W"):
                    row = f.readline()
                if tmp != []:
                    splitted_queries.append(tmp)
                tmp = []
            else:
                row = re.sub(r'[^a-zA-Z\s]+', '', row)
                row = row.upper()
                tmp += row.split()
        if tmp != []:
            splitted_queries.append(tmp)

    queries = []
    for query in splitted_queries:
        queries.append(' '.join(query))

    relevances = []  
    with open('./data/med/MED.REL', 'r') as f:
        tmp = []
        firstrow = [int(x) for x in f.readline().split()]
        current = firstrow[0]
        tmp.append(firstrow[2])
        for row in f:
            row = [int(x) for x in row.split()]
            if row[0] == current:
                tmp.append(row[2])
            else:
                relevances.append(tmp)
                tmp = []
                current = row[0]
                tmp.append(row[2])

    return articles, splitted_articles, queries, splitted_queries, relevances

def import_CRAN():
    """
    This function imports the content of ../data/cran/ folder.

    num of articles: 1398
    num of queries: 225

    Returns:
        articles: (list) List of strings. articles[i-1] is the i-th article of cran.all.1400 file.
        splitted_articles: (list) List of lists. splitted_articles[i-1] is the i-th article of cran.all.1400 file, splitted in words.
        queries: (list) List of strings. queries[i-1] is the i-th query of cran.qry file.
        splitted_queries: (list) List of lists. splitted_queries[i-1] is the i-th query of cran.qry file, splitted in words.
        relevances: (list) List of lists. relevances[i] is the list of relevant documents for query in queries[i].
    """
    splitted_articles = []
    with open('./data/cran/cran.all.1400', 'r') as f:
        tmp = []
        for row in f:
            if row.startswith(".I"):
                while not row.startswith(".W"):
                    row = f.readline()
                if tmp != []:
                    splitted_articles.append(tmp)
                tmp = []
            else:
                row = re.sub(r'[^a-zA-Z\s]+', '', row)
                row = row.upper()
                tmp += row.split()
        if tmp != []:
            splitted_articles.append(tmp)

    articles = []
    for doc in splitted_articles:
        articles.append(' '.join(doc))

    splitted_queries = []
    with open('./data/cran/cran.qry', 'r') as f:
        tmp = []
        for row in f:
            if row.startswith(".I"):
                while not row.startswith(".W"):
                    row = f.readline()
                if tmp != []:
                    splitted_queries.append(tmp)
                tmp = []
            else:
                row = re.sub(r'[^a-zA-Z\s]+', '', row)
                row = row.upper()
                tmp += row.split()
        if tmp != []:
            splitted_queries.append(tmp)

    queries = []
    for query in splitted_queries:
        queries.append(' '.join(query))

    relevances = []
    with open('./data/cran/cran.rel', 'r') as f:
        tmp = []
        firstrow = [int(x) for x in f.readline().split()]
        current = firstrow[0]
        tmp.append(firstrow[1])
        for row in f:
            row = [int(x) for x in row.split()]
            if row[0] == current:
                tmp.append(row[1])
            else:
                relevances.append(tmp)
                tmp = []
                current = row[0]
                tmp.append(row[1])
        relevances.append(tmp)

    return articles, splitted_articles, queries, splitted_queries, relevances

