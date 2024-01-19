# DeepSemanticHashing
Repository for the project of Information Retrieval @ University of Trieste held by professor Laura Nenzi, DSSC Course, A.Y. 2023/2024.

## Outline of the project
This project aims to implement different models for Latent Semantic Analysis (LSA) using deep neural networks and SVD technique, and compare their performances.
The goal of the assessment is to evaluate the quality of the learnt latent space.
The following papers were used as starting point to implement the code:
- [Neural Variational Inference for Text Processing](https://arxiv.org/abs/1511.06038)
- [Unsupervised Neural Generative Semantic Hashing](https://arxiv.org/abs/1906.00671)
- [Semantic Hashing](https://www.sciencedirect.com/science/article/pii/S0888613X08001813)

The datasets used can be distinguished in two different categories:
- dataset inside [data](./data/) folder are small dataset, which comprises:
    - corpus of documents/articles (`.ALL` file)
    - a set of provided queries (`.QRY` file)
    - for each query, the corresponding relevance judgements (`.REL` file)
- [20newsgroup](http://qwone.com/~jason/20Newsgroups/) dataset, a large dataset consisting of 20 different classes of documents, divided by topic. The dataset is already divided into training and test set, but not contains queries and relevance judgments, hence the assessment is performed with different techinques.

## Structure of the repository
- [data](./data/) folder contains the following dataset:
    - time dataset (423 documents, 83 queries)
    - cran dataset (1398 documents, 225 queries)
    - med dataset (1033 documents, 29 queries)
- [src](./src/) folder contains the source code:
    - [autoencoder.py](./src/autoencoder.py) implementation from scratch of autoencoder architecture
    - [variational_autoencoder.py](./src/variational_autoencoder.py) implementation from scratch of variational autoencoder architecture
    - [IR class](./src/IR.py) implementation of a python class from scratch useful for the analysis
    - [import_dataset.py](./src/import_dataset.py) code for read and import data stored inside [data](./data/) folder.
    - [pipelines.py](./src/pipelines.py) implementation of two pipelines for the analysis of the two different categories of dataset.
    - [utils.py](./src/utils.py) code for the implementation of some useful functions.

- [out](./out/) folder contains `.txt` outputs from the analysis done on [data](./data/).
- [data_analysis](./data_analysis.ipynb) notebook for the analysis of the [data](./data/)'s datasets.
- [20news_analysis](./20news_analysis.ipynb) notebook for the analysis of the [20newsgroup](http://qwone.com/~jason/20Newsgroups/) dataset.

## Procedure for [data](./data/) analysis
The following pipeline is followed to perform the analysis:
1. Build the embedding matrix of the corpus, with both word-count and tf-idf embeddings

2. Build the SVD latent space using the corpus
3. Process the queries using the SVD latent space
4. Retrieve the documents using the SVD latent space

5. Train a AutoEncoder model using the embedding matrix
6. Build the AutoEncoder latent space using the corpus
7. Process the queries using the AutoEncoder latent space, i.e. given the .QRY queries, retrieve the documents using the AutoEncoder latent space. 

8. Train a VariationalAutoEncoder model using the embedding matrix
9. Build the VariationalAutoEncoder latent space using the corpus
10. Process the queries using the VariationalAutoEncoder latent space, i.e. given the .QRY queries, retrieve the documents using the VariationalAutoEncoder latent space.

11. Compute the precision and recall for each method

## Procedure for [20newsgroup](http://qwone.com/~jason/20Newsgroups/) analysis
The following pipeline is followed to perform the analysis:
1. import the data
2. Use a CountVectorizer to build the bag of words matrix of the training data
3. Build the trainlaoder using the embdedded training data
4. Train a AutoEncoder model using the trainloader
5. Build the latent space using the training data
6. For each document in the test set:
    - compute its latent representation
    - compute the cosine similarity between the latent representation and the training documents
    - retrieve the top k documents with highest similarity and the k nearest neighbors
    - compare the labels of the retrieved documents with the label of the test document
    - compute the precision for both methods
7. Average the precision over all the test documents



## Results
In this section are reported the results obtained from the analysis of the different datasets.

### [20newsgroup](http://qwone.com/~jason/20Newsgroups/) dataset
The following table reports the results obtained from the analysis of the [20newsgroup](http://qwone.com/~jason/20Newsgroups/) dataset. The model used is the AutoEncoder, trained on word-count embedding matrix of the training data.
It was tested with an increasing number of classses, and with different latent dimensions (50 and 200). Then, cosine similarity and nearest neighbours were used to retrieve the documents, and compared following step 6-7 of the procedure for [20newsgroup](http://qwone.com/~jason/20Newsgroups/) analysis.

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



### Datasets in [data](./data/)
The following table reports the results obtained from the analysis of the datasets in [data](./data/). The model used were SVD, the AutoEncoder and the VariationalAutoEncoder.
Both word-count and tf-idf embeddings were tested for each model, and the results are reported in the table below.
The percentage % represent the percentage of queries within the .QRY file that have at least one relevant document in the top 15 retrieved documents. The precision P and recall R are computed on the top 15 retrieved documents. $\ell$ is the average loss computed for AE and VAE.

|           |SVD        |AE         |VAE         |
|-----------|-----------|-----------|------------|
|TIME - WC|83%, P: 15%, R: 65%|71%, P: 13%, R: 49%, $\ell$=0.095|20%, P: 1.2%, R: 7%, $\ell$=0.113|
|TIME - TFIDF|85%, P: 18%, R: 72%|58%, P: 8%, R: 32%, $\ell$=0.0019|1%, P: 0.6%, R:2.3%, $\ell$=0.003|
|CRAN - WC|66%, P: 9%, R: 19%|52%, P: 5%, R: 12%, $\ell$=0.009|7%, P: 0.05%, R: 1%, $\ell$=0.011|
|CRAN - TFIDF|73%, P: 12%, R: 25%|38%, P: 4%, R: 7%, $\ell$=0.016|11%, P: 0.07%, R: 1.3%, $\ell$=0.017|
|MED - WC|93%, P: 50%, R: 35%|93%, P: 42%, R: 29%, $\ell$=0.42|17%, P: 1%, R: 1%, $\ell$=0.43|
|MED - TFIDF|96%, P: 68%, R: 48%|96%, P: 42%, R: 30%, $\ell$=0.11|38%, P: 3%, R: 2%, $\ell$=0.12|

