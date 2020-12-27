import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, target, target2=None, transform=None):
        self.data = [torch.Tensor(d).long() for d in data]
        self.target = torch.Tensor(target).long()
        self.target2 = target2
        if self.target2 is not None:
            self.target2 = torch.Tensor(self.target2).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        if self.target2 is not None:
            yy = self.target2[index]
            return x, y, yy
        else:
            return x, y

    def __len__(self):
        return len(self.data)
