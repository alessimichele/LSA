# DeepSemanticHashing
Repository for the project of Information Retrieval

## Outline of the project
This project aims to implement different models for Latent Semantic Analysis (LSA) using deep neural networks and SVD technique, and comapre their performances.
All assessments goal is to evaluate the quality of the learnt latent space.
The following papers were used as starting point to implement the code:
- miao16
- neural...
- 

The datasets used can be distinguished in two different categories:
- dataset inside [data](./data/) folder are small dataset, which comprises:
    - corpus of documents/articles (`.ALL` file)
    - a set of provided queries (`.QRY` file)
    - for each query, the corresponding relevance judgements (`.REL` file)
- [20newsgroup](http://qwone.com/~jason/20Newsgroups/) dataset, a large dataset consisting of 20 different classes of documents, divided by topic. The dataset is already divided into training and test set, but not contains queries and relevance judgments, hence the assessment is performed with different techinques.

## Structure of the repository
- [data](./data/) contains the following dataset:
    - time dataset
    - cran dataset
    - med dataset
- [src](./src/) contains the source code:
    - [autoencoder.py](./src/autoencoder.py) implementation from scratch of autoencoder architecture
    - [variational_autoencoder.py](./src/variational_autoencoder.py) implementation from scratch of variational autoencoder architecture
    - [IR class](./src/IR.py) implementation of a python class from scratch useful for the analysis
    - [import_dataset.py] code for read and import data stored inside [data](./data/) folder.

- [data_analysis](./data_analysis.ipynb) notebook for the analysis of the [data](./data/)'s datasets.
- [20news_analysis](./20news_analysis.ipynb) notebook for the analysis of the [20newsgroup](http://qwone.com/~jason/20Newsgroups/) dataset.

## Procedure for [data](./data/) analysis
The following pipeline is followed to perform the analysis:
1. import the data
2. vectorize the corpus using different embeddings, namely word-count embedding and tf-idf embedding
3. train the model on the corpus to learn the latent space
4. assess the model using the queries

## Procedure for [20newsgroup](http://qwone.com/~jason/20Newsgroups/) analysis
The following pipeline is followed to perform the analysis:
1. import the data
2. vectorize the corpus using word-count embedding
3. train the model on the trainingset to learn the latent space
4. assess the model using the testset: for each document in the testset, project it in the latent space and pick the most similar document in the trainingset. Then, compare the label of the most similar document with the label of the document in the testset. 


## Results
### Datasets in [data](./data/)

|           |SVD        |AE         |VAE         |
|-----------|-----------|-----------|------------|
|TIME - WC|83%, P: 15%, R: 65%|71%, P: 13%, R: 49%, $\ell$=0.095|20%, P: 1.2%, R: 7%, $\ell$=0.113|
|TIME - TFIDF|85%, P: 18%, R: 72%|58%, P: 8%, R: 32%, $\ell$=0.0019|1%, P: 0.6%, R:2.3%, $\ell$=0.003|
|CRAN - WC|66%, P: 9%, R: 19%|52%, P: 5%, R: 12%, $\ell$=0.009|7%, P: 0.05%, R: 1%, $\ell$=0.011|
|CRAN - TFIDF|73%, P: 12%, R: 25%|38%, P: 4%, R: 7%, $\ell$=0.016|11%, P: 0.07%, R: 1.3%, $\ell$=0.017|
|MED - WC|93%, P: 50%, R: 35%|93%, P: 42%, R: 29%, $\ell$=0.42|17%, P: 1%, R: 1%, $\ell$=0.43|
|MED - TFIDF|96%, P: 68%, R: 48%|96%, P: 42%, R: 30%, $\ell$=0.11|38%, P: 3%, R: 2%, $\ell$=0.12|

### [20newsgroup](http://qwone.com/~jason/20Newsgroups/) dataset

|# classess        |latent dimension         | Accuracy cosine similarity   | Accuracy nearest neighbours|
|------------------|-------------------------|------------------------------|----------------------------|
|2            |50                        |    85%        | 83% |
|2            |200                       |       94%     |   93%  |
|4            |50                       |       54%     |   26% | 
|4            |200                       |      67%      |  59% |
|6            |50                       |       47%     |  17% | 
|6            |200                       |      54%      | 40% |
|8            |50                       |      50%     |  45%   |
|8            |200                       |        52%    | 40%|


## AGGIUNGERE

METTERE GLI OUT.TXT
METTERE CODICE SORGENTE PER LE PIPELINE
METTERE QUALCOSA SUL SALVATAGGIO DEI MODELLI VOLENDO
METTERE PAPER & SPIEGARE MEGLIO
METTERE PAPER IMPLEMENTATION E DIRE CHE NON VIENE SEMMAI