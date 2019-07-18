import torch


class LogisticRegression(torch.nn.Module):
    """
    Multinomial  logistic regression model for classification
    """

    def __init__(self, input_dim: int, n_classes: int):
        """
        Initialize a logistic regression model

        Arguments:
            input_dim: the dimension of the embedding input
            n_classes: number of classes for classification task

        Return:
             None
        """

        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, n_classes)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Linear transformation

        Arguments:
            batch: input batch of embeddings

        Returns:
            predicted probabilities for the N classes
        """

        y_pred_logits = self.linear(batch)
        return y_pred_logits

