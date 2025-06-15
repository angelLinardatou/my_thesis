from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """Create PyTorch Dataset for tokenized data."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        """Return dataset size."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Return single data sample."""
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
 
