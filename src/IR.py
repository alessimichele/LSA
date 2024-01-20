"""
Code for Information Retrieval course project @ University of Trieste, MSc in Data Science & Scientific Computing A.Y. 2023/2024.
Author: Michele Alessi

This file contains the code implementation of IR class, a class that allows embedding the corpus in different latent spaces and compute the similarity between the queries in .QRY file and the corpus in the latent space.
"""

import torch
import numpy as np
from datetime import date

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity 

from src.utils import compute_pre_rec, custom_preprocessor
from src.autoencoder import AutoEncoder, train as train_ae, build_train_loader
from src.variational_autoencoder import VariationalAutoEncoder, train as train_vae


class IR():
    def __init__(self, name, corpus, splitted_corpus, queries, splitted_queries, relevances):
        """
        Args:
            name: (str) Name of the IR object.
            corpus: (list) List of strings. Each string is a document in the corpus.
            splitted_corpus: (list) List of lists. Each list is a document in the corpus splitted into words.
            queries: (list) List of strings. Each string is a query in the .QRY file.
            splitted_queries: (list) List of lists. Each list is a query in the .QRY file splitted into words.
            relevances: (list) List of lists. Each list is a list of relevant documents for the query in queries[i].

        Attributes:
            cname: (str) Name of the IR object.
            file: (file) File where to write down the results.
            len_corpus: (int) Number of documents in the corpus.
            len_queries: (int) Number of queries in the .QRY file.
            
            Embedding attributes:
            tfidf_vectorizer: (skleran.feature_extraction.text.TfidfVectorizer object) TfidfVectorizer object used to build the tfidf matrix.
            tfidf_matrix: (scipy.sparse.csr.csr_matrix) TfidfVectorizer.fit_transform(corpus) output matrix.
            wc_vectorizer: (skleran.feature_extraction.text.CountVectorizer object) CountVectorizer object used to build the wordcount matrix.
            wc_matrix: (scipy.sparse.csr.csr_matrix) CountVectorizer.fit_transform(corpus) output matrix.

            matrix: (scipy.sparse.csr.csr_matrix) Updated by select_matrix_type(), it is the actual embedding matrix in use to build the latent space.
            matrix_type: (str) Updated by select_matrix_type(), it is the name of the matrix ('tfidf' or 'wc')
            __security: (bool) Security attribute, it is used to avoid the user to change the matrix (avoid errors like building the latent space with a matrix and computing the similarity with another one).
            
            SVD latent space attributes:
            svd: (sklearn.decomposition._truncated_svd.TruncatedSVD object) Functional.
            svd_latent_space: (numpy.ndarray) Output matrix representing the latent space of the corpus built with SVD.
            svd_latent_space_computed: (bool) True if the latent space has been computed.
            n_components: (int) Latent space dimension for SVD.

            Autoencoder latent space attributes:
            autoencoder: (src.autoencoder.AutoEncoder object) Model.
            trained: (bool) True if the autoencoder has been trained.
            autoencoder_latent_space: (numpy.ndarray) Output matrix representing the latent space of the corpus built with the trained autoencoder.
            autoencoder_latent_space_computed: (bool) True if the latent space has been computed.
            losses: (list) List of losses for each iteration during the autoencoder training.
            z_dim: (int) Latent space dimension for Autoencoder.

            Variational Autoencoder latent space attributes:
            vautoencoder: (src.variational_autoencoder.VariationalAutoEncoder object) Model.
            vtrained: (bool) True if the VAE has been trained.
            vautoencoder_latent_space: (numpy.ndarray) Output matrix representing the latent space of the corpus built with the trained variational autoencoder.
            vautoencoder_latent_space_computed: (bool) True if the latent space has been computed.
            vlosses: (list) List of losses for each iteration during the variational autoencoder training.
            vz_dim: (int) Latent space dimension for VAE.

            Queries attributes:
            similarity_matrix_svd: (numpy.ndarray) Output matrix representing the similarity between all the queries in .QRY file and the corpus in the SVD latent space.
            similarity_matrix_autoencoder: (numpy.ndarray) Output matrix representing the similarity between all the queries in .QRY file and the corpus in the Autoencoder latent space.
            similarity_matrix_vautoencoder: (numpy.ndarray) Output matrix representing the similarity between all the queries in .QRY file and the corpus in the Variational Autoencoder latent space.

        Methods:
            __str__: (str) Print the general information about the IR object.
            __getattr__: (str) Print an error message if the attribute is not found.
            __getattribute__: (str) Print an error message if the attribute is not found.

            Embedding methods:
            build_tfidf_matrix: (None) Build the tfidf matrix.
            build_wc_matrix: (None) Build the wordcount matrix.
            select_matrix_type: (None) Select the matrix type to use.

            SVD methods:
            build_svd_latent_space: (None) Build the latent space using svd decomposition.
            process_query_svd: (numpy.ndarray) Compute the similarity between the queries in .QRY file and the corpus in the SVD latent space.

            Autoencoder methods:
            build_autoencoder: (None) Declare the autoencoder architecture.
            train_autoencoder: (None) Train the autoencoder.
            autoencoder_purge: (None) Delete the autoencoder.
            build_autoencoder_latent_space: (None) Build the latent space using the trained autoencoder.
            process_query_autoencoder: (numpy.ndarray) Compute the similarity between the queries in .QRY file and the corpus in the Autoencoder latent space.
            
            save_model: (None) Save the trained model.

            Variational Autoencoder methods:
            build_variational_autoencoder: (None) Declare the variational autoencoder architecture.
            train_variational_autoencoder: (None) Train the variational autoencoder.
            variational_autoencoder_purge: (None) Delete the variational autoencoder.
            build_variational_autoencoder_latent_space: (None) Build the latent space using the trained variational autoencoder.
            process_query_vautoencoder: (numpy.ndarray) Compute the similarity between the queries in .QRY file and the corpus in the Variational Autoencoder latent space.
            
            complete_queries_info: (None) Print all the information about the queries in .QRY file (not working for a single query).
        """


        self.cname = name
        self.file = open(f"./out/{self.cname}.txt", "a")
        self.file.write(f"Author: Michele Alessi\n")
       
        today = date.today()
        self.file.write(f"Date: {today}\n\n")

        # corpus attributes
        self.corpus = corpus
        self.len_corpus = len(corpus)
        self.splitted_corpus = splitted_corpus
        self.queries = queries
        self.len_queries = len(queries)
        self.splitted_queries = splitted_queries
        self.relevances = relevances

############## corpus vectorization ##############
        # matrix attributes
        self.tfidf_vectorizer = None # skleran.feature_extraction.text.TfidfVectorizer object
        self.tfidf_matrix = None # TfidfVectorizer.fit_transform(corpus) output matrix
        self.wc_vectorizer = None # skleran.feature_extraction.text.CountVectorizer object
        self.wc_matrix = None # CountVectorizer.fit_transform(corpus) output matrix

        self.matrix = None # updated by select_matrix_type(), it is the actual matrix
        self.matrix_type = None # updated by select_matrix_type(), it is the name of the matrix
        self.__security = False # security attribute, it is used to avoid the user to change the matrix

############# Latent space  ##############
        # latent space attributes: SVD
        self.svd = None # functional
        self.svd_latent_space = None # output matrix representing the latent space of the corpus
        self.svd_latent_space_computed = False # True if the latent space has been computed
        self.n_components = None # latent space dimension for SVD

        # latent space attributes: Autoencoder
        self.autoencoder= None # model
        self.trained = False # True if the autoencoder has been trained
        self.autoencoder_latent_space = None # output matrix representing the latent space of the corpus
        self.autoencoder_latent_space_computed = False # True if the latent space has been computed
        self.lossess = None # list of losses for each epoch
        self.z_dim = None # latent space dimension for Autoencoder

        # latent space attributes: Variational Autoencoder 
        self.vautoencoder = None  # model
        self.vtrained = False  # True if the VAE has been trained
        self.vautoencoder_latent_space = None  # output matrix representing the latent space of the corpus
        self.vautoencoder_latent_space_computed = False  # True if the latent space has been computed
        self.vlossess = None  # list of losses for each epoch
        self.vz_dim = None  # latent space dimension for VAE

############## Queries attributes ##############
        self.similarity_matrix_svd = None # output matrix representing the similarity between all the queries in .QRY file and the corpus in the SVD latent space
        self.similarity_matrix_autoencoder = None # output matrix representing the similarity between all the queries in .QRY file and the corpus in the Autoencoder latent space
        self.similarity_matrix_vautoencoder = None  # output matrix representing the similarity between all the queries in .QRY file and the corpus in the VAE latent space



    def __str__(self):
        """
        Print the general information about the IR object.
        """
        status = (
            f"General information:\n"
            f"IR object with {self.len_corpus} documents and {self.len_queries} queries.\n")
        
        mat_info = (f"Current matrix-type (i.e. text-embedding type: tfidf, wc) in use is: {self.matrix_type}.\n"
            f"Matrix shape: {self.matrix.shape} ---> (# docs: {self.matrix.shape[0]}, # words: {self.matrix.shape[1]})\n\n"
            if self.matrix is not None else ""
        )
            
        svd_info = (
            f"SVD information:\n"
            f"SVD latent space: {self.svd_latent_space_computed}\n"
            f"SVD latent space dimension: {self.n_components}\n\n"
            if self.svd_latent_space_computed else ""
        )
        
        ae_info = ("Autoencoder information:\n"
            f"Autoencoder trained: {self.trained}\n"
            f"Autoencoder latent space: {self.autoencoder_latent_space_computed}\n"
            f"Autoencoder latent space dimension: {self.z_dim}\n"
            f"Autoencoder architecture: {self.autoencoder}\n"
            f"Losses: {self.lossess[-1]}\n" 
            if self.trained else ""
        )
            
        vae_info = ("Variational Autoencoder information:\n"
            f"Variational Autoencoder trained: {self.vtrained}\n"
            f"Variational Autoencoder latent space: {self.vautoencoder_latent_space_computed}\n"
            f"Variational Autoencoder latent space dimension: {self.vz_dim}\n"
            f"Variational Autoencoder architecture: {self.vautoencoder}\n"    
            f"Losses: {self.vlossess[-1]}\n"
            if self.vtrained else ""
        )

        qinfo = (
            f"\nQueries:\n"
            f"Similiraty with .QRY queries computed in SVD latent space under {self.matrix_type} text embedding: {self.similarity_matrix_svd is not None}\n"
            f"Similarity with .QRY queries computed in Autoencoder latent space under {self.matrix_type} text embedding: {self.similarity_matrix_autoencoder is not None}\n "
            f"Similarity with .QRY queries computed in Variational Autoencoder latent space under {self.matrix_type} text embedding: {self.similarity_matrix_vautoencoder is not None}\n"
            "To have complete information about .QRY queries, call complete_queries_info()\n"
        )

        possible_actions = (
            f"\nPossible actions:\n"
            f"1. Build tfidf matrix: build_tfidf_matrix()\n"
            f"2. Build wc matrix: build_wc_matrix()\n"
            f"3. Select matrix type: select_matrix_type('tfidf' or 'wc')\n"
            f"4. Build SVD latent space: build_svd_latent_space(n_components)\n"
            f"5. Build autoencoder: build_autoencoder(input_dim, h_dim_1, h_dim_2, h_dim_3, z_dim)\n"
            f"6. Train autoencoder: train_autoencoder(BATCH_SIZE, NUM_EPOCHS, loss_fn, lr)\n"
            f"7. Build autoencoder latent space: build_autoencoder_latent_space()\n"
            f"8. Process query in SVD latent space: process_query_svd(query)\n"
            f"9. Process query in Autoencoder latent space: process_query_autoencoder(query)\n"
            f"10. Retrieve documents: retrieve_documents(similarity_matrix, n_docs)\n"
        )
        return status + mat_info + svd_info + ae_info + vae_info + qinfo + possible_actions
    
    def complete_queries_info(self, write_down = False):
        """
        Print all the information about the queries in .QRY file (not working for a single query)
        """
        q_info = ""
        if self.similarity_matrix_svd is not None:
            result_svd, precision_svd, recall_svd = self.retrieve_documents(self.similarity_matrix_svd, verbose=False)
            q_info += "\nSVD info:\n"
            q_info += "\n".join([f"Query {i}: P: {precision_svd[i]} R: {recall_svd[i]}" for i in range(len(self.similarity_matrix_svd))])
            if write_down:
                self.file.write(f"SVD queries info:\n")
                self.file.write("for each query, the top 15 documents retrieved are listed in the form: doc_id, score\n\n")
                self.file.write("query_id, doc_id, score\n")
                for i, dict in enumerate(result_svd):
                    for j, score in dict.items():
                        self.file.write(f"{i+1}, {j}, {score}\n")
                self.file.write("\n\n")

        if self.similarity_matrix_autoencoder is not None:
            result_autoencoder, precision_autoencoder, recall_autoencoder = self.retrieve_documents(self.similarity_matrix_autoencoder, verbose=False)
            q_info += "\nAutoEncoder info:\n"
            q_info += "\n".join([f"Query {i}: P: {precision_autoencoder[i]} R: {recall_autoencoder[i]}" for i in range(len(self.similarity_matrix_autoencoder))])
            if write_down:
                self.file.write(f"AutoEncoder queries info:\n")
                self.file.write("for each query, the top 15 documents retrieved are listed in the form: doc_id, score\n\n")
                self.file.write("query_id, doc_id, score\n")
                for i, dict in enumerate(result_autoencoder):
                    for j, score in dict.items():
                        self.file.write(f"{i+1}, {j}, {score}\n")
                self.file.write("\n\n")
        if self.similarity_matrix_vautoencoder is not None:
            result_vautoencoder, precision_vautoencoder, recall_vautoencoder = self.retrieve_documents(self.similarity_matrix_vautoencoder, verbose=False)
            q_info += "\nVariationalAutoEncoder info:\n"
            q_info += "\n".join([f"Query {i}: P: {precision_vautoencoder[i]} R: {recall_vautoencoder[i]}" for i in range(len(self.similarity_matrix_vautoencoder))])
            if write_down:
                self.file.write(f"VariationalAutoEncoder queries info:\n")
                self.file.write("for each query, the top 15 documents retrieved are listed in the form: doc_id, score\n\n")
                self.file.write("query_id, doc_id, score\n")
                for i, dict in enumerate(result_vautoencoder):
                    for j, score in dict.items():
                        self.file.write(f"{i+1}, {j}, {score}\n")
                self.file.write("\n\n")

        if self.similarity_matrix_svd is None and self.similarity_matrix_autoencoder is None and self.similarity_matrix_vautoencoder is None:
            raise ValueError("No similarity matrix computed yet. Call process_query_svd() or process_query_autoencoder() or process_query_vautoencoder() first.")
        
        print("Queries information:\n", q_info)
        print("\n\nAverage precision and recall:\n")
        if self.similarity_matrix_svd is not None:
            print(f"SVD: P: {np.mean(precision_svd)} R: {np.mean(recall_svd)}")
        if self.similarity_matrix_autoencoder is not None:
            print(f"AutoEncoder: P: {np.mean(precision_autoencoder)} R: {np.mean(recall_autoencoder)}")
        if self.similarity_matrix_vautoencoder is not None:
            print(f"VariationalAutoEncoder: P: {np.mean(precision_vautoencoder)} R: {np.mean(recall_vautoencoder)}")

 
    def select_matrix_type(self, matrix_type):
        """
        Select the matrix type to use (tfidf or wc).

        Args:
            matrix_type: (str) 'tfidf' or 'wc'
        """
        if self.__security is False:
            if matrix_type == "tfidf":
                if self.tfidf_matrix is None:
                    raise ValueError("Tfidf matrix not built yet. Call build_tfidf_matrix() first.")
                self.matrix = self.tfidf_matrix
                self.matrix_type = "tfidf"
                self.__security = True
                print("Tfidf matrix selected.\n")
            elif matrix_type == "wc":
                if self.wc_matrix is None:
                    raise ValueError("Wordcount matrix not built yet. Call build_wc_matrix() first.")
                self.matrix = self.wc_matrix
                self.matrix_type = "wc"
                self.__security = True
                print("Wordcount matrix selected.\n")
            else:
                raise ValueError("Matrix type not supported.")
            
            self.file.write(f"Current matrix-type (i.e. text-embedding type: tfidf, wc) in use is: {self.matrix_type}.\n")
            self.file.write(f"Matrix shape: {self.matrix.shape} ---> (# docs: {self.matrix.shape[0]}, # words: {self.matrix.shape[1]})\n\n")

        else:
            raise ValueError(f"Matrix type already selected: {self.matrix_type}.\n"
                              f"For code-reliability reason it is not possible to change it. If you want to, create a new IR object.")


    def __getattr__(self, name):
        return f"Attribute '{name}' not found."

    def __getattribute__(self, name):
        return super().__getattribute__(name)

############## tfidf and wordcount matrices ##############
    def build_tfidf_matrix(self, vectorizer = TfidfVectorizer(preprocessor=custom_preprocessor, stop_words='english')):
        """
        Fit the tfidf vectorizer on the corpus and build the tfidf matrix.

        Args:
            vectorizer: (sklearn.feature_extraction.text.TfidfVectorizer object) TfidfVectorizer object used to build the tfidf matrix.
        """
        print("Building tfidf matrix...\n")
        self.tfidf_matrix = vectorizer.fit_transform(self.corpus)
        self.tfidf_vectorizer = vectorizer
        print("Done.\n")

    def build_wc_matrix(self, vectorizer = CountVectorizer(preprocessor=custom_preprocessor, stop_words='english')):
        """
        Fit the wordcount vectorizer on the corpus and build the wordcount matrix.

        Args:
            vectorizer: (sklearn.feature_extraction.text.CountVectorizer object) CountVectorizer object used to build the wordcount matrix.
        """
        print("Building wc matrix...\n")
        self.wc_matrix = vectorizer.fit_transform(self.corpus)
        self.wc_vectorizer = vectorizer
        print("Done.\n")

############## SVD latent space ##############
    def build_svd_latent_space(self, n_components=100):
        """
        Fit the TruncatedSVD on the matrix and build the latent space.

        Args:
            n_components: (int) Latent space dimension for SVD.
        """

        if self.matrix is None:
            raise ValueError("Matrix not set yet. Call select_matrix_type() first.")
        print("Building latent space using svd decomposition...\n")
        print(f"Current matrix in use is: {self.matrix_type}.\n")
        
        self.svd = TruncatedSVD(n_components=n_components)
        self.svd_latent_space = self.svd.fit_transform(self.matrix)
        self.n_components = n_components # latent space dimension for SVD
        self.svd_latent_space_computed = True
        print("Done.\n")

    # Queries similarity in SVD space 
    def process_query_svd(self, query=None):
        """
        Process the query in the SVD latent space.

        Args:
            query: single string, or None, if None the similarity between all the queries and the corpus will be computed

        Returns:
            result: output matrix representing the similarity between the query (resp. all the queries) and the corpus in the SVD latent space
        """
        if self.svd_latent_space is None:
            raise ValueError("SVD latent space not built yet. Call build_svd_latent_space() first to find the projection matrix.\n" 
                             "Then process_query_svd() will use that matrix to project the queries in the latent space.")
        print(f"Computing query-corpus similarity using SVD latent space...\n"
              f"Embedding type: {self.matrix_type}\n"
              f"Latent SVD dimension: {self.n_components}\n")
        if self.matrix_type == "tfidf":
            if query is not None and type(query) == str:
                queries_transformed = self.tfidf_vectorizer.transform([query])
            elif query is None:
                queries_transformed = self.tfidf_vectorizer.transform(self.queries)
            else:
                raise ValueError("Query must be either a string or None (in this case the similarity with all the available queries will be computed).")
        elif self.matrix_type == "wc":
            if query is not None and type(query) == str:
                queries_transformed = self.wc_vectorizer.transform([query])
            elif query is None:
                queries_transformed = self.wc_vectorizer.transform(self.queries)
            else:
                raise ValueError("Query must be either a string or None (in this case the similarity with all the available queries will be computed).")
        else:
            raise ValueError("Matrix type not declared. Call select_matrix_type() first.")
        
        queries_reduced = self.svd.transform(queries_transformed)
        similarity_matrix_svd = cosine_similarity(queries_reduced, self.svd_latent_space) 
        if query is None:
            self.similarity_matrix_svd = similarity_matrix_svd
        
        print("Done.\n")
        return similarity_matrix_svd


############## Autoencoder latent space ##############
    def build_autoencoder(self, input_dim, h_dim_1, h_dim_2, z_dim):
        """
        Call to the AutoEncoder class to build the autoencoder model.

        Args:
            input_dim: (int) Number of neurons in the input layer.
            h_dim_1: (int) Number of neurons in the first hidden layer.
            h_dim_2: (int) Number of neurons in the second hidden layer.
            z_dim: (int) Number of neurons in the latent layer.
        """
        print("Declaring autoencoder architecture...\n")
        self.autoencoder= AutoEncoder(input_dim=input_dim, h_dim_1=h_dim_1, h_dim_2=h_dim_2, z_dim=z_dim)
        self.z_dim = z_dim  # latent space dimension for Autoencoder
        print("Done.\n")

    def train_autoencoder(self, BATCH_SIZE, NUM_EPOCHS, loss_fn, lr ):
        """
        Call to the build_train_loader() function to build the train_loader.
        Call to the train() function to train the autoencoder model.

        Args:
            BATCH_SIZE: (int) Batch size for the train_loader.
            NUM_EPOCHS: (int) Number of epochs for the train_loader.
            loss_fn: (torch.nn.modules.loss) Loss function to use. 
            lr: (float) Learning rate.
        """
        if self.matrix is None:
            raise ValueError("Matrix not set yet. Call select_matrix_type() first.")
        if self.autoencoder is None:
            raise ValueError("Autoencoder not built yet. Call build_autoencoder() first.") 
        print(f"Current matrix in use is: {self.matrix_type}.\n")
        print("Training autoencoder ...\n")
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader = build_train_loader(self.matrix.toarray(), BATCH_SIZE)
        self.lossess = train_ae(NUM_EPOCHS=NUM_EPOCHS, train_loader=train_loader, model=self.autoencoder, loss_fn=loss_fn, lr=lr, DEVICE=DEVICE, INPUT_DIM=self.matrix.shape[1], scheduler=None)
        self.trained = True
        print("Done.\n")

    def autoencoder_purge(self):
        """
        Delete the autoencoder model.
        """
        if self.autoencoder is not None:
            del self.autoencoder
            self.autoencoder = None
            self.trained = False
            self.autoencoder_latent_space = None
            self.autoencoder_latent_space_computed = False
            self.lossess = None
            self.z_dim = None
            print("Model purged successfully.\n")
        else:
            print("Model not present.\n")

        
    def save_model(self, path):
        """
        Save the trained autoencoder model.

        Args:
            path: (str) Path where to save the model.
        """
        if self.trained is False:
            raise ValueError("Autoencoder not trained yet, cannot save a model not trained. Call train_autoencoder() first.")
        torch.save(self.autoencoder.state_dict(), path)
        print("Model saved successfully.")
        

    def build_autoencoder_latent_space(self):
        """
        Build the latent space using the trained autoencoder on the matrix.

        This function iterates over the matrix and for each document it applies the autoencoder to the document vector, and picks the output of the latent layer as the latent representation of the document.
        """
        if self.autoencoder is None:
            raise ValueError("Autoencoder not built yet. Call build_autoencoder() first.")
        if not self.trained:
            raise ValueError("Autoencoder not trained yet. Call train_autoencoder() first.")
        print("Building latent space using autoencoder...\n")
        print(f"Current matrix in use is: {self.matrix_type}.\n")
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.autoencoder.eval()
        # per ogni doc del corpus, ne faccio la proiezione nel latent space usando il trained autoencoder
        torchmat = torch.tensor(self.matrix.toarray()).float()
        latent_space = np.zeros((self.len_corpus, self.z_dim))
        for i, vec in enumerate(torchmat):
            print("Progress: {:.2f}%".format((i+1)/self.len_corpus*100), end="\r")
            vec = vec.to(DEVICE)
            _, z = self.autoencoder(vec)
            latent_space[i] = z.detach().numpy()

        self.autoencoder_latent_space = latent_space
        self.autoencoder_latent_space_computed = True
        print("Done.\n")

    # Queries similarity in Autoencoder space 
    def process_query_autoencoder(self, query=None):
        """
        Iterate over the queries and for each query it applies the autoencoder to the query vector, and picks the output of the latent layer as the latent representation of the query.
        Then it computes the similarity between the queries and the corpus in the Autoencoder latent space.

        Args:
            query: single string, or None, if None the similarity between all the queries and the corpus will be computed
            result: output matrix representing the similarity between all the queries and the corpus in the Autoencoder latent space

        Returns:
            result: output matrix representing the similarity between the query (resp. all the queries) and the corpus in the Autoencoder latent space
        """
        
        if self.trained is False:
            raise ValueError("Autoencoder not trained yet. Call train_autoencoder() first, then use the model to project the queries in the latent space.")
        print(f"Computing single query-corpus similarity using AutoEncoder latent space...\n"
              f"Embedding type: {self.matrix_type}\n"
              f"Latent AutoEncoder dimension: {self.z_dim}\n")
        if self.matrix_type == "tfidf":
            if query is not None and type(query) == str:
                queries_transformed = self.tfidf_vectorizer.transform([query])
            elif query is None:
                queries_transformed = self.tfidf_vectorizer.transform(self.queries)
            else:
                raise ValueError("Query must be either a string or None (in this case the similarity with all the available queries will be computed).")
        elif self.matrix_type == "wc":
            if query is not None and type(query) == str:
                queries_transformed = self.wc_vectorizer.transform([query])
            elif query is None:
                queries_transformed = self.wc_vectorizer.transform(self.queries)
            else:
                raise ValueError("Query must be either a string or None (in this case the similarity with all the available queries will be computed).")
        else:
            raise ValueError("Matrix type not declared. Call select_matrix_type() first.")
        

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        queries_transformed = queries_transformed.toarray()
        #print(queries_transformed.shape)
        queries_reduced = np.zeros((queries_transformed.shape[0], self.z_dim))
        self.autoencoder.eval()
        # per ogni q, faccio la proiezione nel latent
        for i, vec in enumerate(queries_transformed):
            vec = torch.tensor(vec).float().to(DEVICE)
            _, z = self.autoencoder(vec)
            queries_reduced[i] = z.detach().numpy()


        similarity_matrix_autoencoder = cosine_similarity(queries_reduced, self.autoencoder_latent_space)

        if query is None:
            self.similarity_matrix_autoencoder = similarity_matrix_autoencoder
        
        print("Done.\n")
        return similarity_matrix_autoencoder


 ############## Variational Autoencoder latent space ##############
    def build_variational_autoencoder(self, input_dim, h_dim_1, h_dim_2,  z_dim):
        """
        Call to the VariationalAutoEncoder class to build the variational autoencoder model.

        Args:
            input_dim: (int) Number of neurons in the input layer.
            h_dim_1: (int) Number of neurons in the first hidden layer.
            h_dim_2: (int) Number of neurons in the second hidden layer.
            z_dim: (int) Number of neurons in the latent layer.
        """
        print("Declaring variational autoencoder architecture...\n")
        self.vautoencoder = VariationalAutoEncoder(input_dim=input_dim, h_dim_1=h_dim_1, h_dim_2=h_dim_2, z_dim=z_dim)
        self.vz_dim = z_dim  # latent space dimension for VAE
        print("Done.\n")

    def train_variational_autoencoder(self, BATCH_SIZE, NUM_EPOCHS, loss_fn, lr, scheduler = None):
        """
        Call to the build_train_loader() function to build the train_loader.
        Call to the train() function to train the variational autoencoder model.

        Args:
            BATCH_SIZE: (int) Batch size for the train_loader.
            NUM_EPOCHS: (int) Number of epochs for the train_loader.
            loss_fn: (torch.nn.modules.loss) Loss function to use.
            lr: (float) Learning rate.
            scheduler: (torch.optim.lr_scheduler) Learning rate scheduler.
        """
        if self.matrix is None:
            raise ValueError("Matrix not set yet. Call select_matrix_type() first.")
        if self.vautoencoder is None:
            raise ValueError("Variational Autoencoder not built yet. Call build_variational_autoencoder() first.") 
        print(f"Current matrix in use is: {self.matrix_type}.\n")
        print("Training variational autoencoder ...\n")
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader = build_train_loader(self.matrix.toarray(), BATCH_SIZE)
        self.vlossess = train_vae(NUM_EPOCHS=NUM_EPOCHS, train_loader=train_loader, model=self.vautoencoder, loss_fn=loss_fn, lr=lr, DEVICE=DEVICE, INPUT_DIM=self.matrix.shape[1], scheduler=scheduler)
        self.vtrained = True
        print("Done.\n")

    def variational_autoencoder_purge(self):
        """
        Delete the variational autoencoder model.
        """
        if self.vautoencoder is not None:
            del self.vautoencoder
            self.vautoencoder = None
            self.vtrained = False
            self.vautoencoder_latent_space = None
            self.vautoencoder_latent_space_computed = False
            self.vlossess = None
            self.vz_dim = None
            print("Model purged successfully.\n")
        else:
            print("Model not present.\n")

    def build_variational_autoencoder_latent_space(self):
        """
        Build the latent space using the trained variational autoencoder on the matrix.

        This function iterates over the matrix and for each document it applies the variational autoencoder to the document vector, and picks the output of the latent layer as the latent representation of the document.

        Note: the latent space is built by taking the mean of the latent representation of the document over 10 iterations.
        """


        if self.vautoencoder is None:
            raise ValueError("Variational Autoencoder not built yet. Call build_variational_autoencoder() first.")
        if not self.vtrained:
            raise ValueError("Variational Autoencoder not trained yet. Call train_variational_autoencoder() first.")
        print("Building latent space using variational autoencoder...\n")
        print(f"Current matrix in use is: {self.matrix_type}.\n")
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vautoencoder.eval()
        torchmat = torch.tensor(self.matrix.toarray()).float()
        latent_space = np.zeros((self.len_corpus, self.vz_dim))
        for i, vec in enumerate(torchmat):
            print("Progress: {:.2f}%".format((i+1)/self.len_corpus*100), end="\r")
            vec = vec.to(DEVICE)
            for _ in range(10):
                _,_,_, z = self.vautoencoder(vec)
            # take the mean
                latent_space[i] += z.detach().numpy()
            latent_space[i] /= 10
            #latent_space[i] = z.detach().numpy()

        self.vautoencoder_latent_space = latent_space
        self.vautoencoder_latent_space_computed = True
        print("Done.\n")

    # Queries similarity in Variational Autoencoder space 
    def process_query_vautoencoder(self, query=None):
        """
        Iterate over the queries and for each query it applies the variational autoencoder to the query vector, and picks the output of the latent layer as the latent representation of the query.
        Then it computes the similarity between the queries and the corpus in the Variational Autoencoder latent space.

        Args:
            query: single string, or None, if None the similarity between all the queries and the corpus will be computed

        Returns:
            result: output matrix representing the similarity between the query (resp. all the queries) and the corpus in the Variational Autoencoder latent space
        """
        if self.vtrained is False:
            raise ValueError("Variational Autoencoder not trained yet. Call train_variational_autoencoder() first, then use the model to project the queries in the latent space.")
        print(f"Computing single query-corpus similarity using Variational Autoencoder latent space...\n"
              f"Embedding type: {self.matrix_type}\n"
              f"Latent Variational AutoEncoder dimension: {self.vz_dim}\n")
        if self.matrix_type == "tfidf":
            if query is not None and type(query) == str:
                queries_transformed = self.tfidf_vectorizer.transform([query])
            elif query is None:
                queries_transformed = self.tfidf_vectorizer.transform(self.queries)
            else:
                raise ValueError("Query must be either a string or None (in this case the similarity with all the available queries will be computed).")
        elif self.matrix_type == "wc":
            if query is not None and type(query) == str:
                queries_transformed = self.wc_vectorizer.transform([query])
            elif query is None:
                queries_transformed = self.wc_vectorizer.transform(self.queries)
            else:
                raise ValueError("Query must be either a string or None (in this case the similarity with all the available queries will be computed).")
        else:
            raise ValueError("Matrix type not declared. Call select_matrix_type() first.")
        

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        queries_transformed = queries_transformed.toarray()
        queries_reduced = np.zeros((queries_transformed.shape[0], self.vz_dim))
        self.vautoencoder.eval()
        for i, vec in enumerate(queries_transformed):
            vec = torch.tensor(vec).float().to(DEVICE)
            _,_,_, z = self.vautoencoder(vec)
            queries_reduced[i] = z.detach().numpy()

        
        similarity_matrix_vautoencoder = cosine_similarity(queries_reduced, self.vautoencoder_latent_space)

        if query is None:
            self.similarity_matrix_vautoencoder = similarity_matrix_vautoencoder
        
        print("Done.\n")
        return similarity_matrix_vautoencoder



############## Retrieve docs given a similarity matrix ##############
    def retrieve_documents(self, similarity_matrix, n_docs=15, verbose = True):
        """
        Retrieve the top n_docs documents for each query.

        Args:
            similarity_matrix: output matrix representing the similarity between all the queries (or the query) and the corpus in the latent space
            n_docs: (int) Number of documents to retrieve for each query.
            verbose: (bool) If True, print the precision and recall for each query.

        Returns:
            result: list of dictionaries, each dictionary contains the top n_docs documents for a query
            precision: list of precision for each query (available only if len(similarity_matrix)>1, meaning that the similarity between all the queries and the corpus has been computed)
            recall: list of recall for each query (available only if len(similarity_matrix)>1, meaning that the similarity between all the queries and the corpus has been computed)

        Note: if the query is a string, only the top n_docs documents for that query will be retrieved. Since the query does not belong to the .QRY file, relevences are not available and the precision and recall will not be computed.
        """

        result =[]
        for i in range(len(similarity_matrix)):
            tmp = {}
            # tmp = {doc_id: similarity_score of doc_id with query i}
            for j in similarity_matrix[i].argsort()[-n_docs:][::-1]:
                tmp[j+1] = similarity_matrix[i, j] # j+1 because doc_id starts from 1 in relevances
            result.append(tmp)
        #print('len_result', len(result))
        
        if len(similarity_matrix)>1:
            precision, recall = compute_pre_rec(result, self.relevances)
            count = 0
            for i in range(len(precision)):
                if recall[i] != 0:
                    if verbose:
                        print(f"Query {i}: P: {precision[i]} R: {recall[i]}\n")
                    count += 1
            if verbose:
                print(f"Query at least with positive precision: {count} out of {len(precision)}")
            return result, precision, recall
        else:
            print(f"Retrived docs for the given query:\n",
                  "!WARNING! doc_id starts from 1\n")
            for doc_id, score in result[0].items(): # doc_id starts from 1
                print(f"Doc id: {doc_id} Score: {score}\n")
            return result, None, None
        

    
