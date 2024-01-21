"""
Code for Information Retrieval course project @ University of Trieste, MSc in Data Science & Scientific Computing A.Y. 2023/2024.
Author: Michele Alessi

This file contains the code for two pipelines to test the models:
    - pipeline_20: pipeline for the 20newsgroups dataset
    - data_pipeline: pipeline for the dataset inside ../data/ folder.
"""


import numpy as np
from matplotlib import pyplot as plt
import torch 
from torch.utils.data import DataLoader

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from src.utils import *
from src.import_dataset import *
from src.IR import *
from src.autoencoder import *



def pipeline_20(n_categories, batch_size, num_epochs, z_dim, k):
    """
    Pipeline to test the models on the 20newsgroups dataset.

    1. Load the dataset using n_categories categories
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

    Args:
        n_categories: (int) number of categories to use
        batch_size: (int) batch size for training
        num_epochs: (int) number of epochs for training
        z_dim: (int) dimension of the latent space
        k: (int) top k documents to retrieve

    Returns:
        mean_sim: (float) average precision using cosine similarity
        mean_neigh: (float) average precision using nearest neighbors

    """
    categories = [
 'rec.autos',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'talk.politics.misc',
 'talk.religion.misc']

    categories = categories[:n_categories]
    twenty_train = fetch_20newsgroups(subset='train',
    shuffle=True, random_state=42, categories=categories)
    twenty_test = fetch_20newsgroups(subset='test',
        shuffle=True, random_state=42, categories=categories)
    
    # Define a vectorizer with custom preprocessor 
    vectorizer = CountVectorizer(preprocessor=custom_preprocessor, stop_words='english')


    # Fit and transform the training data
    X_train_counts = vectorizer.fit_transform(twenty_train.data)

    # Transform the test data using the same vectorizer
    X_test_counts = vectorizer.transform(twenty_test.data)

    

    X_train_tensor = torch.Tensor(X_train_counts.toarray())
    y_train_tensor = torch.LongTensor(twenty_train.target)

    X_test_tensor = torch.Tensor(X_test_counts.toarray())
    y_test_tensor = torch.LongTensor(twenty_test.target)



    train_dataset = TextDatasetUnlabeled(X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    #test_dataset = TextDatasetUnlabeled(X_test_tensor)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    model = AutoEncoder(input_dim=X_train_counts.shape[1], h_dim_1=7000, h_dim_2=500, z_dim=z_dim)

    _=train(NUM_EPOCHS=num_epochs, train_loader=train_loader, model= model, loss_fn=nn.BCELoss(), lr=1e-4, DEVICE= torch.device('cuda' if torch.cuda.is_available() else 'cpu'), INPUT_DIM=X_test_counts.shape[1], scheduler=None)


    #build the latent space using the training data
    with torch.no_grad():
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        latent_training_space = np.zeros((len(X_train_counts.toarray()), z_dim))
        model.eval()
        with torch.no_grad():
            for i, doc in enumerate(X_train_tensor):
                doc = doc.to(DEVICE)
                _, z=model(doc)
                latent_training_space[i] = z.detach().numpy()

    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(latent_training_space)

    model.eval()
    precision = []
    with torch.no_grad():
        for i, doc in enumerate(X_test_tensor):
            label = int(y_test_tensor[i])
            doc = doc.to(DEVICE)
            _, z=model(doc)
            z = z.detach().numpy()
            
            # cosine similarity
            similarity = cosine_similarity(z.reshape(1,-1), latent_training_space)
            top_30_sim = similarity.argsort()[0][-k:]
            labels_sim = y_train_tensor[top_30_sim]
            prec_sim = len(labels_sim[labels_sim == label])/k

            # nearest neighbors
            neighbors = neigh.kneighbors(z.reshape(1,-1), return_distance=False)
            labels_neigh = y_train_tensor[neighbors[0]]
            prec_neigh = len(labels_neigh[labels_neigh == label])/k

            precision.append((prec_sim, prec_neigh))
            array = np.array(precision)
            mean_sim = array[:,0].mean()
            mean_neigh = array[:,1].mean()
    return mean_sim, mean_neigh


######################################################################################################################
######################################################################################################################
######################################################################################################################

def data_pipeline(IR_object, embedding='wc', BATCH_SIZE=50, lr = 1e-5, NUM_EPOCHS=100, h_dim_1=5000, h_dim_2 = 1000, z_dim = 200, n_docs = 15):
    """
    Pipeline to test the models on the dataset used in the project.

    1. Build the embedding matrix using the IR_object
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

    Args:
        IR_object: An IR object.
        embedding: (str) 'wc' for word count embedding, 'tfidf' for tfidf embedding.
        BATCH_SIZE: (int) Batch size for training.
        lr: (float) Learning rate for training.
        NUM_EPOCHS: (int) Number of epochs for training.
        h_dim_1: (int) First hidden layer dimension.
        h_dim_2: (int) Second hidden layer dimension.
        z_dim: (int) Latent space dimension.
        n_docs: (int) Number of documents to retrieve.

    Returns:
        result_svd: A list of dictionaries {doc_id: score} for the SVD method, where result_svd[i] is the result for queries[i].
        precision_svd: A list containing the precision values for the SVD method. precision_svd[i] is the precision for queries[i].
        recall_svd: A list containing the recall values for the SVD method. recall_svd[i] is the recall for queries[i].
        result_ae: A list of dictionaries {doc_id: score} for the AutoEncoder method, where result_ae[i] is the result for queries[i].
        precision_ae: A list containing the precision values for the AutoEncoder method. precision_ae[i] is the precision for queries[i].
        recall_ae: A list containing the recall values for the AutoEncoder method. recall_ae[i] is the recall for queries[i].
        losses: A list of loss values for the AutoEncoder method.
        result_vae: A list of dictionaries {doc_id: score} for the VariationalAutoEncoder method, where result_vae[i] is the result for queries[i].
        precision_vae: A list containing the precision values for the VariationalAutoEncoder method. precision_vae[i] is the precision for queries[i].
        recall_vae: A list containing the recall values for the VariationalAutoEncoder method. recall_vae[i] is the recall for queries[i].
        vlossess: A list of loss values for the VariationalAutoEncoder method.
    """
    
    if embedding == 'wc':
        IR_object.build_wc_matrix()
        IR_object.select_matrix_type('wc')
    elif embedding == 'tfidf':
        IR_object.build_tfidf_matrix()
        IR_object.select_matrix_type('tfidf')
    else:
        print('Invalid embedding')
        return
    
    print('SVD\n\n')
    IR_object.build_svd_latent_space(n_components=100)
    _ = IR_object.process_query_svd()
    result_svd, precision_svd, recall_svd = IR_object.retrieve_documents(IR_object.similarity_matrix_svd, n_docs=n_docs)

    print('AutoEncoder\n\n')
    if embedding == 'wc':
        IR_object.build_autoencoder(input_dim=IR_object.matrix.shape[1], h_dim_1=h_dim_1, h_dim_2=h_dim_2, z_dim=z_dim)
        IR_object.train_autoencoder(BATCH_SIZE=BATCH_SIZE, NUM_EPOCHS=NUM_EPOCHS, loss_fn=torch.nn.BCELoss(), lr=lr)
    else:
        IR_object.build_autoencoder(input_dim=IR_object.matrix.shape[1], h_dim_1=h_dim_1, h_dim_2=h_dim_2, z_dim=z_dim)
        IR_object.train_autoencoder(BATCH_SIZE=BATCH_SIZE, NUM_EPOCHS=NUM_EPOCHS, loss_fn=torch.nn.MSELoss(), lr=lr)


    window = IR_object.matrix.shape[0]//BATCH_SIZE

    plt.plot( [sum(IR_object.lossess[i:i+window])/window for i in range(0, len(IR_object.lossess), window)])

    IR_object.build_autoencoder_latent_space()
    _ = IR_object.process_query_autoencoder()
    result_ae, precision_ae, recall_ae = IR_object.retrieve_documents(IR_object.similarity_matrix_autoencoder, n_docs=n_docs)

    print('VariationalAutoEncoder\n\n')
    if embedding == 'wc':
        IR_object.build_variational_autoencoder(input_dim=IR_object.matrix.shape[1], h_dim_1=h_dim_1, h_dim_2=h_dim_2, z_dim=z_dim)
        IR_object.train_variational_autoencoder(BATCH_SIZE=BATCH_SIZE, NUM_EPOCHS=NUM_EPOCHS, loss_fn=torch.nn.BCELoss(), lr=lr)
    else:
        IR_object.build_variational_autoencoder(input_dim=IR_object.matrix.shape[1], h_dim_1=h_dim_1, h_dim_2=h_dim_2, z_dim=z_dim)
        IR_object.train_variational_autoencoder(BATCH_SIZE=BATCH_SIZE, NUM_EPOCHS=NUM_EPOCHS, loss_fn=torch.nn.MSELoss(), lr=lr)

   
    plt.plot( [sum(IR_object.vlossess[i:i+window])/window for i in range(0, len(IR_object.vlossess), window)])
    IR_object.build_variational_autoencoder_latent_space()
    _ = IR_object.process_query_vautoencoder()
    result_vae, precision_vae, recall_vae = IR_object.retrieve_documents(IR_object.similarity_matrix_vautoencoder, n_docs=n_docs)

    IR_object.complete_queries_info(write_down = True)

    return result_svd, precision_svd, recall_svd, result_ae, precision_ae, recall_ae, IR_object.losses, result_vae, precision_vae, recall_vae, IR_object.vlossess
