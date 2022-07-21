import csvutils
import numpy as np
import matplotlib.pyplot as plt


def draw_line(r0, c0, r1, c1):
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1-c0) < abs(r1-r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = draw_line(c0, r0, c1, r1)
        return (yy, xx, val)

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return draw_line(r1, c1, r0, c0)

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = np.arange(c0, c1+1, dtype=float)
    y = x * (r1-r0) / (c1-c0) + (c1*r0-c0*r1) / (c1-c0)

    valbot = np.floor(y)-y+1
    valtop = y-np.floor(y)

    return (np.concatenate((np.floor(y), np.floor(y)+1)).astype(int), np.concatenate((x,x)).astype(int),
            np.concatenate((valbot, valtop)))

def csvToImage(data3D,data2D,sampleID, width=32, height=32):
    img = np.zeros((width,height,3))

    labels = list()
    for label in data3D["label"]:

      tokens = label.split('_')

      if (len(tokens)==2):
       labels.append(tokens[1])

      if (len(tokens)==3):
       labels.append(tokens[1]+'_'+tokens[2])

    print("Labels ",labels)

    for label in labels:
       xLabel = "2DX_"+label
       yLabel = "2DY_"+label
       zLabel = "3DZ_"+label
       if (xLabel in data2D["label"]) and ( yLabel in data2D["label"]):

         idxX = data2D["label"].index(xLabel)
         idxY = data2D["label"].index(yLabel)
         idxZ = data3D["label"].index(zLabel)

         x2D  = int(width*data2D["body"][sampleID][idxX])
         y2D  = int(height*data2D["body"][sampleID][idxY])
         z3D  = int(data3D["body"][sampleID][idxZ])

         print(xLabel,",",yLabel," => ", x2D, y2D, z3D)
         img[y2D][x2D][0] = int(z3D * 255 / (-400))

    return img


if __name__ == "__main__":
    pose2d=csvutils.readCSVFile("exp/datasets/cmubvh/2d_body_all.csv",memPercentage=100)
    pose3d=csvutils.readCSVFile("exp/datasets/cmubvh/3d_body_all.csv",memPercentage=100)

    poses = 100 
    res = 150


    for p in range(poses):

      img = csvToImage(pose3d, pose2d, p, res, res)

      fig = plt.imshow(img)
      # fig.axes.get_xaxis().set_visible(False)
      # fig.axes.get_yaxis().set_visible(False)
      plt.savefig(f'debug/pose{p}.png')
      plt.cla()
