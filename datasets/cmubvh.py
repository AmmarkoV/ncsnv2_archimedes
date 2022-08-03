import time
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class CMUBVH(Dataset):
    def __init__(self, path, train=True, split=0.05, transform=None):
        bmuvhrows = 1505224
        testrows = int(bmuvhrows * split)

        skiprows=0
        nrows = testrows
        if train:
            skiprows = nrows
            nrows = bmuvhrows - skiprows

        self.data = pd.read_csv(path, nrows=nrows, skiprows=skiprows)

    def __len__(self):
        return self.data.shape[0] 

    def __getitem__(self, idx):
        return self.data.iloc[idx].values


if __name__ == "__main__":
    path = "exp/datasets/cmubvh/3d_body_all.csv"

    dataset = CMUBVH(path, train=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for idx, batch in enumerate(dataloader):
        print(f'idx: {idx}')