from torch.utils.data import Dataset
import torch


class TextDataset(Dataset):

    def __init__(self, data, labels):
        self.data = torch.Tensor(data)
        self.labels = torch.LongTensor(labels)
        assert len(data) == len(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {'data': self.data[index], 'label': self.labels[index]}
