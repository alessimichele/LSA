"""
Code for Information Retrieval course project @ University of Trieste, MSc in Data Science & Scientific Computing A.Y. 2023/2024.
Author: Michele Alessi

This file contains the code to define and train the AutoEncoder model.
"""

import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader



class AutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim_1, h_dim_2, z_dim):
        """
        Class to define the AutoEncoder model, with fixed number of hidden layers set to 2.

        Args:
            input_dim: (int) Input dimension.
            h_dim_1: (int) Dimension of the first hidden layer.
            h_dim_2: (int) Dimension of the second hidden layer.
            z_dim: (int) Dimension of the latent space.
        """
        super().__init__()
        self.input_dim = input_dim

        # encoder
        self.img_2hid = nn.Linear(input_dim, h_dim_1)
        self.hid_2hid_en = nn.Linear(h_dim_1, h_dim_2)
        self.hid_2z = nn.Linear(h_dim_2, z_dim)


        # decoder
        self.R = nn.Linear(z_dim,input_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)

    def encode(self, x):
        h1 = self.relu(self.img_2hid(x))
        h2 = self.relu(self.hid_2hid_en(h1))
        z = self.relu(self.hid_2z(h2))
        return z


    def decode(self, z):
        E = self.R(z)
        return self.softmax(-E)


    def forward(self, x):
        z = self.encode(x)
        x_bin = self.decode(z)
        return x_bin, z


def weights_init(m):
    """
    Function to initialize the weights of the model.

    Args:
        m: A nn.Module object.
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

def build_train_loader(matrix, BATCH_SIZE):
    """
    Function to build a the train loader given the embedding matrix.

    Args:
        matrix: A numpy array of shape (n_samples, n_features).
        BATCH_SIZE: (int) A integer indicating the batch size.
    Returns:
        train_loader: A DataLoader object.
    """
    train_dataset = torch.tensor(matrix).float()
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader

def train(NUM_EPOCHS, train_loader, model, loss_fn, lr, DEVICE, INPUT_DIM, scheduler):
    """
    Function to train the model.

    Args:
        NUM_EPOCHS: (int) Number of epochs to train the model.
        train_loader: A DataLoader object.
        model: A AutoEncoder class object.
        loss_fn: A torch loss function. nn.MSELoss() for tfidf, nn.BCELoss() for wc embedding were used.
        lr: (float) Learning rate.
        DEVICE: Device to train the model on. Typically, 'cuda' or 'cpu'.
        INPUT_DIM: (int) Input dimension. This is matrix.shape[1], were matrix is the embedding matrix used to build the train_loader.
        scheduler: A scheduler object. Used to reduce the learning rate when the loss plateaus. If scheduler is None, the learning rate is constant.

    Returns:
        lossess: A list of loss values.

    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = loss_fn
    # Training
    model.train()
    lossess = []
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1} of {NUM_EPOCHS}")
        loop = tqdm(enumerate(train_loader))
        for i, x in loop:
            # forward pass
            x = x.to(DEVICE)
            x = x.view(x.shape[0], INPUT_DIM)
            x_reconstructed, _ = model(x)

            # loss
            reconstruction_loss = loss_fn(x_reconstructed, x)
           
            # backprop
            loss = reconstruction_loss 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

            if scheduler:
                scheduler.step(loss)
                loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

            lossess.append(loss.item())
    return lossess






