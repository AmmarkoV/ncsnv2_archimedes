import datasets.csvutils 
import numpy as np
import matplotlib.pyplot as plt



def getParentList():
 pl=dict()
 #---------------------------------------------------
 pl["hip"]="hip"               #0  
 pl["neck"]="hip"              #1  
 pl["head"]="neck"             #2
 pl["abdomen"]="chest"         #3
 pl["chest"]="hip"             #4
 pl["eye.l"]="head"    #5    
 pl["eye.r"]="head"    #6   
 pl["EndSite_eye.l"]="eye.l"    #5    
 pl["EndSite_eye.r"]="eye.r"    #6   
 pl["rshoulder"]="neck"        #7    
 pl["relbow"]="rshoulder"      #8      
 pl["rhand"]="relbow"          #9
 pl["lshoulder"]="neck"        #10
 pl["lelbow"]="lshoulder"      #11
 pl["lhand"]="lelbow"          #12
 pl["rhip"]="hip"              #13
 pl["rknee"]="rhip"            #14
 pl["rfoot"]="rknee"           #15
 pl["toe1-2.r"]="rfoot"#16
 pl["toe5-3.r"]="rfoot"#17
 pl["EndSite_toe1-2.r"]="toe1-2.r"#16
 pl["EndSite_toe5-3.r"]="toe5-3.r"#17
 pl["lhip"]="hip"              #18
 pl["lknee"]="lhip"            #19
 pl["lfoot"]="lknee"           #20
 pl["toe1-2.l"]="lfoot"#21
 pl["toe5-3.l"]="lfoot"#22
 pl["EndSite_toe1-2.l"]="toe1-2.l"#21
 pl["EndSite_toe5-3.l"]="toe5-3.l"#22
 return pl
#---------------------------------------------------

parentList = getParentList()


def draw_line(r0, c0, r1, c1):
    #print("draw_line(",r0,c0,r1,c1,")")
    if (r0==r1) and (c0==c1):
       return r0,c0,1 
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

    return (np.concatenate((np.floor(y), np.floor(y)+1)).astype(int), np.concatenate((x,x)).astype(int),np.concatenate((valbot, valtop)))

def trapez(y,y0,w):
    return np.clip(np.minimum(y+1+w/2-y0, -y+1+w/2+y0),0,1)

def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf):
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1-c0) < abs(r1-r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)
        return (yy, xx, val)

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax)

    # The following is now always < 1 in abs
    slope = (r1-r0) / (c1-c0)

    # Adjust weight by the slope
    w *= np.sqrt(1+np.abs(slope)) / 2

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = np.arange(c0, c1+1, dtype=float)
    y = x * slope + (c1*r0-c0*r1) / (c1-c0)

    # Now instead of 2 values for y, we have 2*np.ceil(w/2).
    # All values are 1 except the upmost and bottommost.
    thickness = np.ceil(w/2)
    yy = (np.floor(y).reshape(-1,1) + np.arange(-thickness-1,thickness+2).reshape(1,-1))
    xx = np.repeat(x, yy.shape[1])
    vals = trapez(yy, y.reshape(-1,1), w).flatten()

    yy = yy.flatten()

    # Exclude useless parts and those outside of the interval
    # to avoid parts outside of the picture
    mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))

    return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])

def getJointCoordinates(
                        joint2DLabelList,
                        joint2DBodyList,
                        #----------------
                        joint3DLabelList,
                        joint3DBodyList,
                        #----------------
                        label,
                        #----------------
                        width,
                        height,
                        #----------------
                        sampleID
                       ):
       xLabel = "2DX_"+label
       yLabel = "2DY_"+label
       zLabel = "3DZ_"+label
       if (xLabel in joint2DLabelList) and ( yLabel in joint2DLabelList):

         idxX = joint2DLabelList.index(xLabel)
         idxY = joint2DLabelList.index(yLabel)
         idxZ = joint3DLabelList.index(zLabel)
        
         if (len(joint2DBodyList.shape)==1):
          x2D  = int(min(width-1 ,width*joint2DBodyList[idxX]))
          y2D  = int(min(height-1,height*joint2DBodyList[idxY]))
          z3D  = int(joint3DBodyList[idxZ])
         else:
          x2D  = int(min(width-1 ,width*joint2DBodyList[sampleID][idxX]))
          y2D  = int(min(height-1,height*joint2DBodyList[sampleID][idxY]))
          z3D  = int(joint3DBodyList[sampleID][idxZ])
        
         #print("getJointCoordinates ",xLabel,",",yLabel," => ", x2D, y2D, z3D)
         valueToColor = min(255,int(z3D * 255 / (-400)))
         #print("  val ", valueToColor)
         return x2D,y2D,valueToColor
       #print("getJointCoordinates could not find ",xLabel,",",yLabel," ")
       return 0,0,0



def csvToImage(data3D,data2D,sampleID, width=32, height=32):
    img = np.zeros((width,height,3),dtype=np.uint8)

    labels = list()
    #Gather all labels from our 3D data
    for label in data3D["label"]:
      tokens = label.split('_')
      if (len(tokens)==2):
       labels.append(tokens[1])
      if (len(tokens)==3):
       labels.append(tokens[1]+'_'+tokens[2])
    #print("Labels ",labels)

    for label in labels:
       x2D,y2D,val        = getJointCoordinates(data2D["label"],data2D["body"],data3D["label"],data3D["body"],label,width,height,sampleID)
       xP2D,yP2D,Pval     = getJointCoordinates(data2D["label"],data2D["body"],data3D["label"],data3D["body"],parentList[label],width,height,sampleID)
       
       if (x2D!=0) and (y2D!=0) and (xP2D!=0) and (yP2D!=0):
        #Horrible hack to not get out of bounds
        x2D  = min(width-2,x2D)
        y2D  = min(height-2,y2D)
        xP2D = min(width-2,xP2D)
        yP2D = min(height-2,yP2D)
        #---------------------------
        
        y,x,r = draw_line(y2D,x2D,yP2D,xP2D)
        if (type(x)==int):
         #img[y][x][0] = int(r*255)
         #img[y][x][1] = int(r*255)
         img[y][x][2] = int(r*255)
        else:
         for i in range(0,len(y)):
           #img[y[i]][x[i]][0] = int(r[i]*255)
           #img[y[i]][x[i]][1] = int(r[i]*255)
           img[y[i]][x[i]][2] = int(r[i]*255)

       img[y2D][x2D][0]   = val
       img[y2D][x2D][1]   = val
       #img[y2D][x2D][2]   = val
       img[yP2D][xP2D][0] = Pval
       img[yP2D][xP2D][1] = Pval
       #img[yP2D][xP2D][2] = Pval

    return img


if __name__ == "__main__":
    poses      = 100 
    resolution = 100#150

    pose2d=csvutils.readCSVFile("exp/datasets/cmubvh/2d_body_all.csv",memPercentage=poses)
    pose3d=csvutils.readCSVFile("exp/datasets/cmubvh/3d_body_all.csv",memPercentage=poses)

    for p in range(poses):
      print("Dumping pose ",p)
      img = csvToImage(pose3d, pose2d, p, resolution, resolution)

      fig = plt.imshow(img)
      # fig.axes.get_xaxis().set_visible(False)
      # fig.axes.get_yaxis().set_visible(False)
      plt.savefig(f'debug/pose{p}.png')
      plt.cla()
