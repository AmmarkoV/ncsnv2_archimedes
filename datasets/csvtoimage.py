import datasets.csvutils as csvutils 
# import csvutils
import numpy as np
import matplotlib.pyplot as plt

def getParentList():
 pl=dict()
 #---------------------------------------------------
 pl["hip"]="hip"                  #0  
 pl["neck"]="hip"                 #1  
 pl["head"]="neck"                #2
 pl["abdomen"]="chest"            #3
 pl["chest"]="hip"                #4
 pl["eye.l"]="head"               #5    
 pl["eye.r"]="head"               #6   
 pl["EndSite_eye.l"]="eye.l"      #5    
 pl["EndSite_eye.r"]="eye.r"      #6   
 pl["rshoulder"]="neck"           #7    
 pl["relbow"]="rshoulder"         #8      
 pl["rhand"]="relbow"             #9
 pl["lshoulder"]="neck"           #10
 pl["lelbow"]="lshoulder"         #11
 pl["lhand"]="lelbow"             #12
 pl["rhip"]="hip"                 #13
 pl["rknee"]="rhip"               #14
 pl["rfoot"]="rknee"              #15
 pl["toe1-2.r"]="rfoot"           #16
 pl["toe5-3.r"]="rfoot"           #17
 pl["EndSite_toe1-2.r"]="toe1-2.r"#16
 pl["EndSite_toe5-3.r"]="toe5-3.r"#17
 pl["lhip"]="hip"                 #18
 pl["lknee"]="lhip"               #19
 pl["lfoot"]="lknee"              #20
 pl["toe1-2.l"]="lfoot"           #21
 pl["toe5-3.l"]="lfoot"           #22
 pl["EndSite_toe1-2.l"]="toe1-2.l"#21
 pl["EndSite_toe5-3.l"]="toe5-3.l"#22
 return pl
#---------------------------------------------------
parentList = getParentList()
#---------------------------------------------------


#---------------------------------------------------
#---------------------------------------------------
#---------------------------------------------------
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

#---------------------------------------------------
#---------------------------------------------------
#---------------------------------------------------
def getJoint2DCoordinates(
                        joint2DLabelList,
                        joint2DBodyList,
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
       if (xLabel in joint2DLabelList) and ( yLabel in joint2DLabelList):

         idxX = joint2DLabelList.index(xLabel)
         idxY = joint2DLabelList.index(yLabel)
        
         if (len(joint2DBodyList.shape)==1):
          x2D  = int(min(width-1 ,width*joint2DBodyList[idxX]))
          y2D  = int(min(height-1,height*joint2DBodyList[idxY]))
         else:
          x2D  = int(min(width-1 ,width*joint2DBodyList[sampleID][idxX]))
          y2D  = int(min(height-1,height*joint2DBodyList[sampleID][idxY]))

         return x2D,y2D
       #print("getJoint2DCoordinates could not find ",xLabel,",",yLabel," ")
       return 0,0

def getJoint3DCoordinates(
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
       #We reuse the code for 2D coordinates to reduce code surface!
       x2D, y2D = getJoint2DCoordinates(
                                        joint2DLabelList,
                                        joint2DBodyList,
                                        #----------------
                                        label,
                                        #----------------
                                        width,
                                        height,
                                        #----------------
                                        sampleID
                                       ) 
       #We extract the 3D value and return it!
       zLabel = "3DZ_"+label
       if (zLabel in joint3DLabelList):
         idxZ = joint3DLabelList.index(zLabel)
        
         if (len(joint2DBodyList.shape)==1):
          z3D  = int(joint3DBodyList[idxZ])
         else:
          z3D  = int(joint3DBodyList[sampleID][idxZ])

         return x2D, y2D, z3D
       #print("getJoint3DCoordinates could not find ",xLabel,",",yLabel," ")
       return 0,0,0

#---------------------------------------------------
#---------------------------------------------------
#---------------------------------------------------

"""
  Convert a Depth Value to RGB 
"""
def convertDepthValueToRGB(depthValue,near=0,far=650):
   #https://sites.google.com/site/brainrobotdata/home/depth-image-encoding
   #https://developers.google.com/depthmap-metadata/encoding
   
   #Make sure the value is positive 
   depthValue=abs(depthValue)
  
   dNorm = (depthValue - near) / (far-near) 
   d16 = int(dNorm * 65535)
   #Split Unsigned Short
   upper8bits = d16 >> 8
   lower8bits = d16 & 0b0000000011111111
   #-------------------------------------
   r=int(0)
   g=int(upper8bits)
   b=int(lower8bits)
   #-------------------------------------
   #print("Value %0.2f | R=%u G=%u B=%u "%(depthValue,r,g,b))
   #print("Upper {:08b}".format(upper8bits))
   #print("Lower {:08b}".format(lower8bits))
   #print("16bit {:016b}".format(d16))
   return r,g,b

"""
  Convert an RGB encoded value back to the Depth Value 
"""
def convertRGBValueToDepth(r,g,b,near=0,far=650):
   #Make sure value is positive
   # r 
   upper8bits=int(g)
   lower8bits=int(b)

   depthValue = (upper8bits << 8) | lower8bits;
   
   depthValue = depthValue / 65536
   depthValue = near + ( depthValue * (far - near) ) 

   return depthValue

"""
  Euclidean 2D Distance
"""
def distance2D(x1,y1,x2,y2):
   return np.sqrt( ((x1-x2)*(x1-x2)) + ((y1-y2)*(y1-y2)) )


"""
  We begin a line from (s)ource sX,sY which has a value sV
  and draw it up to (t)arget tX,tY which has a value tV
  This function for each currentX,currentY value will try to interpolate the value 
  and output something depending on the closest distance. 
"""
def interpolateValue(sX,sY,sV,tX,tY,tV,currentX,currentY):
   distanceToSource = distance2D(sX,sY,currentX,currentY)
   distanceToTarget = distance2D(tX,tY,currentX,currentY)
   distanceFull     = distance2D(sX,sY,tX,tY)

   value = 0.0
   value = value + sV * (distanceToSource/distanceFull)
   value = value + tV * (distanceToTarget/distanceFull)
   return value


def extractListOfLabelsWithoutCoordinates(origin):
    labels = list()
    #Gather all labels from our 3D data
    for label in origin:
      tokens = label.split('_')
      if (len(tokens)==2):
       labels.append(tokens[1])
      if (len(tokens)==3):
       labels.append(tokens[1]+'_'+tokens[2])
    labels = list(set(labels))
    #print("Labels ",labels)
    return labels



#---------------------------------------------------
#---------------------------------------------------
#---------------------------------------------------
def getJoint3DCoordinatesNormalize(
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
  x2D,y2D,val = getJoint3DCoordinates(
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

def csvToImageDigitalEncoding(data3D,data2D,sampleID, width=32, height=32):
    #First failed experiment with zeros!
    #img = np.zeros((3,width,height))
    #Second experiment will use 0.5 as background
    bkg  = np.random.rand()
    img = np.full((3,width,height),bkg)


    labels = extractListOfLabelsWithoutCoordinates(data3D["label"])
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
       x2D,y2D,val = getJoint3DCoordinatesNormalize(data2D["label"],data2D["body"],data3D["label"],data3D["body"],label,width,height,sampleID)
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
#---------------------------------------------------
#---------------------------------------------------
#---------------------------------------------------




"""
  Convert CSV 2D + 3D Data to an RGB Image!
"""
def csvToImage(data3D,data2D,sampleID, width=32, height=32, rnd=False, translationInvariant=True, interpolateDepth=True, bkg=0.5, encoding=False):
    if (encoding):
        return csvToImageDigitalEncoding(data3D,data2D,sampleID,width=width,height=height)

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
        

    labels = extractListOfLabelsWithoutCoordinates(data3D["label"])

    if (len(labels)==0):
      print("Sample ",sampleID," is empty")
      return img

    #Default Alignment
    alignX2D = 0
    alignY2D = 0
    if (translationInvariant):
       x2D, y2D, _ = getJoint3DCoordinates(data2D["label"],data2D["body"],data3D["label"],data3D["body"],"hip",width,height,sampleID)
       alignX2D = (width/2)  - x2D
       alignY2D = (height/2) - y2D 


    for label in labels:
       x2D,y2D,val        = getJoint3DCoordinates(data2D["label"],data2D["body"],data3D["label"],data3D["body"],label,width,height,sampleID)
       r,g,b = convertDepthValueToRGB(val)

       xP2D,yP2D,Pval     = getJoint3DCoordinates(data2D["label"],data2D["body"],data3D["label"],data3D["body"],parentList[label],width,height,sampleID)
 
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
        
        #TODO : 
        # if ( greenExisting < smallerThanNew ) or ( ( greenExisting == smallerThanNew ) and ( blueExisting == smallerThanNew )   ) 
        #  RESPECT DEPTH ORDER 
        y,x,foo = draw_line(y2D,x2D,yP2D,xP2D)
        if (type(y)==float) or (type(y)==int): 
         img[0][y][x] = 255 #Keypoint should be marked! r
         img[1][y][x] = g
         img[2][y][x] = b
        elif (type(x)==float) or (type(x)==int):
         img[0][y][x] = 255 #Keypoint should be marked! r
         img[1][y][x] = g
         img[2][y][x] = b
        else:
         #By default a blue line between joints
         iR = 0
         iG = 0 
         iB = 255
         for i in range(0,len(y)):
           if (interpolateDepth):
             interpolatedValue = interpolateValue(x2D,y2D,val,xP2D,yP2D,Pval,x[i],y[i])
             iR,iG,iB = convertDepthValueToRGB(interpolatedValue)
           img[0][y[i]][x[i]] = iR
           img[1][y[i]][x[i]] = iG
           img[2][y[i]][x[i]] = iB

        #------------------------- 
        img[0][y2D][x2D]   = 255# Keypoint should be marked!r
        img[1][y2D][x2D]   = g
        img[2][y2D][x2D]   = b
        #-------------------------
        #img[0][yP2D][xP2D] = Pval
        #img[1][yP2D][xP2D] = Pval 

    #print("Encode Min/Max R ",np.min(img[0][:][:]),np.max(img[0][:][:]))
    #print("Encode Min/Max G ",np.min(img[1][:][:]),np.max(img[1][:][:]))
    #print("Encode Min/Max B ",np.min(img[2][:][:]),np.max(img[2][:][:]))

    return img

"""
  Convert CSV 2D Data + Image to 3D Data!
""" 
def imageToCSV(data2D, img, sampleID, width=32, height=32, rnd=False, translationInvariant=True, interpolateDepth=True, bkg=0.5, encoding=False):
    #------------------------------------------------
    labels = extractListOfLabelsWithoutCoordinates(data2D["label"])
    #------------------------------------------------
    data3D          = dict() # <- This will get populated 
    data3D["label"] = list() # <- This will get populated 
    data3D["body"]  = list() # <- This will get populated 
    data3D["body"].append(list()) # <- This will get populated 

    print("Decode Min/Max R ",np.min(img[0][:][:]),np.max(img[0][:][:]))
    print("Decode Min/Max G ",np.min(img[1][:][:]),np.max(img[1][:][:]))
    print("Decode Min/Max B ",np.min(img[2][:][:]),np.max(img[2][:][:]))

    #Default Alignment
    alignX2D = 0
    alignY2D = 0
    if (translationInvariant):
       x2D,y2D = getJoint2DCoordinates(data2D["label"],data2D["body"],"hip",width,height,sampleID)
       alignX2D = (width/2)  - x2D
       alignY2D = (height/2) - y2D 


    for label in labels:
    #------------------------------------------------
      x,y     = getJoint2DCoordinates(
                                      data2D["label"],
                                      data2D["body"],
                                      #----------------
                                      label,
                                      #----------------
                                      width,
                                      height,
                                      #----------------
                                      sampleID
                                     )
      x = int(x+alignX2D)
      y = int(y+alignY2D)
      #------------------------------------------------
      r = int(img[0][y][x])
      g = int(img[1][y][x])
      b = int(img[2][y][x])
      #------------------------------------------------
      val = convertRGBValueToDepth(r,g,b)
      print("label ",label," x=",x," y=",y," r=",r," g=",g," b=",b," val=",val)
      #------------------------------------------------
      data3D["label"].append("3DX_%s" % label)
      data3D["body"][0].append(float(x))
      data3D["label"].append("3DY_%s" % label)
      data3D["body"][0].append(float(y))
      data3D["label"].append("3DZ_%s" % label)
      data3D["body"][0].append(float(val)) 
      #------------------------------------------------
    return data3D

#---------------------------------------------------
#---------------------------------------------------
#---------------------------------------------------

if __name__ == "__main__":
    import sys

    poses      = 10 
    resolution = 150 #64#150
    near = 0 
    far  = 650

    legendStepX=list()
    legendStepY=list()
    legendColor=list()
    for depthValue in range(near,far):
       r,g,b = convertDepthValueToRGB(depthValue)
       legendStepX.append(depthValue)
       legendStepY.append(0)
       legendColor.append([r/255,g/255,b/255]) 
       depthValue2 = int(round(convertRGBValueToDepth(r,g,b)))
       if (depthValue2!=depthValue):
          print("Mismatch at Depth ",depthValue," -> R ",r," G ",g," B ",b," -> ",depthValue2)
          sys.exit(0)
    print("All Depth->RGB->Depth conversions are happening successfuly")
    #sys.exit(0)
    plt.scatter(legendStepX,legendStepY, s=1000, color=legendColor)
    #plt.show()
    plt.savefig(f'debug/colormap.png')
    plt.cla()


    pose2d=csvutils.readCSVFile("exp/datasets/cmubvh/2d_body_all.csv",memPercentage=poses)
    pose3d=csvutils.readCSVFile("exp/datasets/cmubvh/3d_body_all.csv",memPercentage=poses)

    #-----------------------------------
    labels = extractListOfLabelsWithoutCoordinates(pose2d["label"])
    #-----------------------------------
    print("Labels 2D ",pose2d["label"])
    print("Labels 3D ",pose3d["label"])
    print("Labels ",labels)

    for p in range(poses):
      print("Dumping pose ",p)
      img         = csvToImage(pose3d,pose2d,p,resolution,resolution)
      recovered3D = imageToCSV(pose2d,img,p,resolution,resolution)

      #print("Labels 3D ",recovered3D["label"])
      for label in labels:
          thisLabel = "3DZ_%s" % label
          if (thisLabel in pose3d["label"]) and (thisLabel in recovered3D["label"]):
            #------------------------------------------------------
            originalIDX    = pose3d["label"].index(thisLabel)
            originalDepth  = abs(pose3d["body"][p][originalIDX])
            #------------------------------------------------------
            recoveredIDX   = recovered3D["label"].index(thisLabel)
            recoveredDepth = recovered3D["body"][0][recoveredIDX]
            #------------------------------------------------------
            print("Depth Discrepancy %s = %f (org %f,rec %f)"%(label,abs(originalDepth-recoveredDepth),originalDepth,recoveredDepth))

      imgSwapped = np.swapaxes(img,0,2)
      imgSwapped = np.swapaxes(imgSwapped,0,1)
      fig = plt.imshow(imgSwapped.astype(np.uint8))


    
      # fig.axes.get_xaxis().set_visible(False)
      # fig.axes.get_yaxis().set_visible(False)
      plt.imsave(f'debug/pose{p}.png',imgSwapped.astype(np.uint8))
      plt.savefig(f'debug/fig{p}.png')
      plt.cla()


