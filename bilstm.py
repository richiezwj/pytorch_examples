import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

np.random.seed(10)


class BiLSTM(nn.Module):
    """
    Bi-directional Long Short Term Memory model used for word sequence classification. 
    Details can be found: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
    """

    def __init__(
        self,
        embedding_model,
        hidden_dim: int,
        label_size: int,
        num_layers: int,
        lstm_dropout: float,
        dropout: float,
    ):
        """
        Initialize a bi-lstm model for word sequence classification task

        Arguments:
            embedding_model: word embedding model
            hidden_dim: dimension of hidden layer
            label_size: number of classes/labels for the classification task
            num_layers: number of lstm layers
            lstm_dropout: dropout rate used for lstm layer
            dropout: dropout rate for fully connected layer

        Return:
            None
        """

        super(BiLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.label_size = label_size

        # initialize embedding layer
        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(embedding_model.weights).type(torch.FloatTensor)
        )
        self.embedding_dim = embedding_model.vector_dim

        # initial bi-lstm architecture
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=lstm_dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(p=dropout)

        # get final label
        self.hidden2label = nn.Linear(hidden_dim * 2, label_size)

    def init_hidden(self, batch_size: int):
        """
        Initialize hidden state and cell state with all zeros
        """

        return (
            Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim)),
            Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim)),
        )

    def forward(
        self, batch: torch.Tensor, lengths: torch.Tensor, device: torch.device
    ) -> (torch.Tensor, torch.Tensor):
        """
        Forward computation for the module

        Arguments:
            batch: batch of sequences with token indices after padding
            lengths: batch of original sequence lengths
            device: device to run forward calculation on
        
        Return:
            probability distribution of prediction results for the batch
            labels after sorting for the batch

        """
        batch_size = batch.size(0)

        hidden = tuple(v.to(device) for v in self.init_hidden(batch_size))

        embeddings = self.embedding(batch)

        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths.cpu().numpy(), batch_first=True
        )

        lstm_out, (h, c) = self.lstm(packed_embeddings, hidden)

        y = self.hidden2label(
            self.dropout(h.transpose(0, 1).contiguous().view(-1, self.hidden_dim * 2))
        )
        log_probs = F.log_softmax(y, dim=1)

        return log_probs

