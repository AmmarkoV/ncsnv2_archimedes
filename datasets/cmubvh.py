import torch
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
        data3D["label"] = list(self.data3d.head(1)) #columns has all data!
        data3D["body"] = self.data3d.iloc[idx].values
        #_____________________________________________
        data2D = dict()
        data2D["label"] = list(self.data2d.head(1)) #columns has all data!
        data2D["body"] = self.data2d.iloc[idx].values
        #_____________________________________________

        print("GetItem(",idx,") =>  data2d[label]=",data2D["label"]," data2d[body]=",data2D["body"]) 
        print("GetItem(",idx,") =>  data3D[label]=",data3D["label"]," data3D[body]=",data3D["body"]) 

        # dummy label -1.
        return torch.tensor(csvToImage(data3D,data2D, idx, self.res, self.res), dtype=torch.float), -1


if __name__ == "__main__":

    dataset = CMUBVH(train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    os.system("rm debug/*.png")
    import numpy as np

    for idx, (batch, _) in enumerate(dataloader):

      batchSwapped = np.swapaxes(batch.squeeze(),0,2)
      batchSwapped = np.swapaxes(batchSwapped,0,1)
      fig = plt.imshow(batchSwapped)
      # fig.axes.get_xaxis().set_visible(False)
      # fig.axes.get_yaxis().set_visible(False)
      plt.savefig(f'debug/pose{idx}.png')
      plt.cla()
