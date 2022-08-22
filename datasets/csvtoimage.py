import datasets.csvutils as csvutils 
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



def getJointCoordinatesNormalize(
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
  x2D,y2D,val = getJointCoordinates(
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
                                   )
  return float(x2D/width),float(y2D/height),float(val/255)



def csvToImageEncoding(data3D,data2D,sampleID, width=32, height=32):
    #First failed experiment with zeros!
    #img = np.zeros((3,width,height))
    #Second experiment will use 0.5 as background
    bkg  = np.random.rand()
    img = np.full((3,width,height),bkg)

    labels = list()
    #Gather all labels from our 3D data
    for label in data3D["label"]:
      tokens = label.split('_')
      if (len(tokens)==2):
       labels.append(tokens[1])
      if (len(tokens)==3):
       labels.append(tokens[1]+'_'+tokens[2])
    labels = list(set(labels))
    #print("Labels ",labels)
 
    if (len(labels)==0):
      print("Sample ",sampleID," is empty")
      return img

    widthPerJoint  = int(width/3)
    heightPerJoint = int(max(1,height/len(labels)))
    #print(widthPerJoint,"/",heightPerJoint)
   
    yS = 0
    for label in labels:
       xS = 0
       x2D,y2D,val = getJointCoordinatesNormalize(data2D["label"],data2D["body"],data3D["label"],data3D["body"],label,width,height,sampleID)
       #---------------------------------------------------
       encX = float(min(255,x2D*255))
       img[:,xS:xS+widthPerJoint,yS:yS+heightPerJoint]=encX
       xS = xS + widthPerJoint
       
       encY = float(min(255,y2D*255))
       img[:,xS:xS+widthPerJoint,yS:yS+heightPerJoint]=encY
       xS = xS + widthPerJoint

       encV = float(min(255,val*255))
       img[:,xS:xS+widthPerJoint,yS:yS+heightPerJoint]=encV
       yS = yS + heightPerJoint
       #print("Label ",label," x=",x2D," y=",y2D," v=",val," => xS=",xS," yS=",yS," encX=",encX," => encY=",encY," encV=",encV)
   
    return img









def csvToImage(data3D,data2D,sampleID, width=32, height=32, rnd=True, translationInvariant=True, bkg=0.5, encoding=True):
    if (encoding):
        return csvToImageEncoding(data3D,data2D,sampleID,width=width,height=height)

    #First failed experiment with zeros!
    #img = np.zeros((3,width,height))
    #Second experiment will use 0.5 as background
    if not rnd:
        img = np.full((3,width,height),bkg)
    else:
        #All pixels random 
        #img = np.random.uniform(low=0.0, high=1.0, size=(3,width,height))
        #Only background pixels random 
        bkg  = np.random.rand()
        bkg2 = np.random.rand()
        img  = np.full((3,width,height),bkg)
        for y in range(0,height):
          img[:,:,y]=float(y*(abs(bkg2-bkg)/height))
        
    labels = list()
    #Gather all labels from our 3D data
    for label in data3D["label"]:
      tokens = label.split('_')
      if (len(tokens)==2):
       labels.append(tokens[1])
      if (len(tokens)==3):
       labels.append(tokens[1]+'_'+tokens[2])
    labels = list(set(labels))
    #print("Labels ",labels)

    if (len(labels)==0):
      print("Sample ",sampleID," is empty")
      return img

    #Default Alignment
    alignX2D = 0
    alignY2D = 0
    if (translationInvariant):
       x2D,y2D,val = getJointCoordinates(data2D["label"],data2D["body"],data3D["label"],data3D["body"],"hip",width,height,sampleID)
       alignX2D = (width/2)  - x2D
       alignY2D = (height/2) - y2D 
     

    for label in labels:
       x2D,y2D,val        = getJointCoordinates(data2D["label"],data2D["body"],data3D["label"],data3D["body"],label,width,height,sampleID)
       xP2D,yP2D,Pval     = getJointCoordinates(data2D["label"],data2D["body"],data3D["label"],data3D["body"],parentList[label],width,height,sampleID)
 
       #Do alignment
       if (x2D!=0) and (y2D!=0) and (xP2D!=0) and (yP2D!=0):
        x2D  = int(x2D  + alignX2D)
        y2D  = int(y2D  + alignY2D)
        xP2D = int(xP2D + alignX2D)
        yP2D = int(yP2D + alignY2D)
       
       if (x2D!=0) and (y2D!=0) and (xP2D!=0) and (yP2D!=0):
        #Horrible hack to not get out of bounds
        x2D  = int(min(width-2 ,x2D ))
        y2D  = int(min(height-2,y2D ))
        xP2D = int(min(width-2 ,xP2D))
        yP2D = int(min(height-2,yP2D))
        #---------------------------
        
        y,x,r = draw_line(y2D,x2D,yP2D,xP2D)
        if (type(y)==float) or (type(y)==int):
         img[2][y][x] = r
        elif (type(x)==float) or (type(x)==int):
         img[2][y][x] = r
        else:
         for i in range(0,len(y)):
           img[2][y[i]][x[i]] = r[i]

        #-------------------------
        img[0][y2D][x2D]   = val
        img[1][y2D][x2D]   = val
        #-------------------------
        #img[0][yP2D][xP2D] = Pval
        #img[1][yP2D][xP2D] = Pval 

    return img


if __name__ == "__main__":
    poses      = 100 
    resolution = 32#150

    pose2d=csvutils.readCSVFile("exp/datasets/cmubvh/2d_body_all.csv",memPercentage=poses)
    pose3d=csvutils.readCSVFile("exp/datasets/cmubvh/3d_body_all.csv",memPercentage=poses)

    for p in range(poses):
      print("Dumping pose ",p)
      img = csvToImage(pose3d, pose2d, p, resolution, resolution)
      
      imgSwapped = np.swapaxes(img,0,2)
      imgSwapped = np.swapaxes(imgSwapped,0,1)
      fig = plt.imshow(imgSwapped)
      # fig.axes.get_xaxis().set_visible(False)
      # fig.axes.get_yaxis().set_visible(False)
      plt.savefig(f'debug/pose{p}.png')
      plt.cla()
