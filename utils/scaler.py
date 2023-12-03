
# Import modules
import torch


class StandardScaler():
    """Standard scaler."""
    def __init__(self):
        """Class initialization (mean 0 and unit std)."""
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        """Fit with the training data.

        Args:
            data (list): list of data values.
        """
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        """Apply data standardization.

        Args:
            data (list): input data.

        Returns:
            list: standardized data.
        """
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        """De-standardize the input data.

        Args:
            data (list): standardized data.

        Returns:
            list: de-standardized data.
        """
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean
