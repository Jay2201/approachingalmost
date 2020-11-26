import torch
from torch import embedding
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, embedding_matrix):
        """
        :param embedding_matrix: numpy array with vectors for all words
        """
        super(LSTM, self).__init__()
        # number of words = number of rows in embedding matrix
        num_words = embedding_matrix.shape[0]

        # dimension of embedding is num of columns in the matrix
        embed_dim = embedding_matrix.shape[1]

        # we define an input embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=num_words,
            embedding_dim=embed_dim
        )

        # embedding matrix is used as weights of
        # the embedding layer
        self.embedding.weight = nn.Parameter(
            torch.tensor(
                embedding_matrix,
                dtype=torch.float32
            )
        )

        # we don't want to train pretrained embeddings
        self.embedding.weight.requires_grad = False

        # a simple bidirectional LSTM with hidden size of 128
        self.lstm = nn.LSTM(
            embed_dim,
            128,
            bidirectional=True,
            batch_first=True,
        )

        # output layer which is linear layer
        # we have only one output
        # input (512) = 128 + 128 for mean and same for max pooling
        self.out = nn.Linear(512, 1)

        
