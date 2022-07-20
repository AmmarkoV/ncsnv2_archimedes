import csvutils
import numpy as np

def csvToImage(data3D,data2D,sampleID):
    img = np.zeros((3,32,32))

    labels = list()
    for i in range(0,data3D["body"].shape[1]):
      label = data3D["label"][i]
      split = label.split('_')
      if (len(split)==2):
       labels.append(split[1])
      if (len(split)==3):
       labels.append(split[1]+'_'+split[2])
    print("Labels ",labels)

    for label in labels:
       xLabel = "2DX_"+label
       yLabel = "2DY_"+label
       zLabel = "3DZ_"+label
       if (xLabel in data2D["label"]) and ( yLabel in data2D["label"]):
         idxX = data2D["label"].index(xLabel)
         idxY = data2D["label"].index(yLabel)
         x2D  = int(32*data2D["body"][sampleID][idxX])
         y2D  = int(32*data2D["body"][sampleID][idxY])
         idxZ = data3D["label"].index(zLabel)
         z3D  = data3D["body"][sampleID][idxZ]
         print(xLabel,",",yLabel," => ", x2D, y2D, z3D)
         img[0][y2D][x2D]=255
    return img


if __name__ == "__main__":
    pose2d=csvutils.readCSVFile("exp/datasets/cmubvh/2d_body_all.csv",memPercentage=100)
    pose3d=csvutils.readCSVFile("exp/datasets/cmubvh/3d_body_all.csv",memPercentage=100)
    csvToImage(pose3d,pose2d,55)
