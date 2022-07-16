import csvutils
from torch.utils.data import Dataset


class CMUBVH(Dataset):
    def __init__(self):

        self.data3d = csvutils.readCSVFile("exp/datasets/cmubvh/3d_body_all.csv")

    def __len__(self):
        return self.data3d["body"].shape[0]

    def __getitem__(self, idx):

        pose = self.data3d["body"][idx]

        return pose


if __name__ == "__main__":
    cmubvh = CMUBVH()

    print(len(cmubvh))