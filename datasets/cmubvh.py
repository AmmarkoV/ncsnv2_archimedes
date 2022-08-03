import time
import os
import pandas as pd
from datasets.csvtoimage import csvToImage
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class CMUBVH(Dataset):
    def __init__(self, train=True, split=0.05, res=32, transform=None):

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
        #_____________________________________________
        data3D = dict()
        data3D["label"] = list(self.data3d.columns)
        data3D["body"] = self.data3d.iloc[idx].values
        #_____________________________________________
        data2D = dict()
        data2D["label"] = list(self.data2d.columns)
        data2D["body"] = self.data2d.iloc[idx].values
        #_____________________________________________
        return csvToImage(data3D,data2D, idx, self.res, self.res)


if __name__ == "__main__":

    dataset = CMUBVH(train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    os.system("rm debug/*.png")

    for idx, batch in enumerate(dataloader):

      fig = plt.imshow(batch.squeeze())
      # fig.axes.get_xaxis().set_visible(False)
      # fig.axes.get_yaxis().set_visible(False)
      plt.savefig(f'debug/pose{idx}.png')
      plt.cla()