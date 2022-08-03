import time
import pandas as pd
from csvtoimage import csvToImage
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class CMUBVH(Dataset):
    def __init__(self, train=True, split=0.05, res=64, transform=None):

        self.res = res
        path2d = "exp/datasets/cmubvh/2d_body_all.csv"
        path3d = "exp/datasets/cmubvh/3d_body_all.csv"

        bmuvhrows = 1505224
        testrows = int(bmuvhrows * split)

        skiprows=0
        nrows = testrows
        if train:
            skiprows = nrows
            nrows = bmuvhrows - skiprows

        self.data2d = pd.read_csv(path2d, nrows=nrows, skiprows=skiprows)
        self.data3d = pd.read_csv(path3d, nrows=nrows, skiprows=skiprows)

    def __len__(self):

        assert(self.data2d.shape[0] == self.data3d.shape[0])
        return self.data2d.shape[0] 

    def __getitem__(self, idx):
        return csvToImage(self.data3d.iloc[idx].values, self.data2d.iloc[idx].values, idx, self.res, self.res)


if __name__ == "__main__":

    dataset = CMUBVH(train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for idx, batch in enumerate(dataloader):
      fig = plt.imshow(batch)
      # fig.axes.get_xaxis().set_visible(False)
      # fig.axes.get_yaxis().set_visible(False)
      plt.savefig(f'debug/pose{p}.png')
      plt.cla()