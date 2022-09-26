#!/usr/bin/env python3
import h5py
import numpy as np
import csv
import os
import sys
from os import listdir
from os.path import isdir, join
#import xml.etree.ElementTree as ET

#from scipy.spatial import procrustes
#from scipy.linalg import orthogonal_procrustes

from align2DPoints import pointListReturnXYZListForScatterPlot,pointListsReturnAvgDistance,compute_similarity_transform
#from procrustes import procrustes_aligned
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d, Axes3D  
print("Using matplotlib:",matplotlib.__version__)

def get3DDistance(jX,jY,jZ,pX,pY,pZ):
 return np.sqrt( ((jX-pX)*(jX-pX)) + ((jY-pY)*(jY-pY)) + ((jZ-pZ)*(jZ-pZ)) )



#-----------------------------------------------------------------------------------------------------------------------
def drawFrameError(averageErrorDistances,averageMotionEstimationDistancesBetweenFrames,ax4):
 
 #minimum=np.min(averageErrorDistances)
 #ax4.plot((0,len(averageErrorDistances)), (minimum,minimum),label='Minimum Error')

 #maximum=np.max(averageErrorDistances)
 #ax4.plot((0,len(averageErrorDistances)), (maximum,maximum),label='Maximum Error')

 #median=np.median(averageErrorDistances)
 #ax4.plot((0,len(averageErrorDistances)), (median,median),label='Median Error (%0.2f mm)'% median)

 average=np.average(averageErrorDistances) 
 ax4.plot((0,len(averageErrorDistances)), (average,average),label='Average of average errors (%0.2f mm)' % average)

 ax4.plot(averageErrorDistances, label='Our method average error in millimeters')
 ax4.plot(averageMotionEstimationDistancesBetweenFrames, label='Distance of average joint from previous frame')

 #ax4.set_ylim(auto=False,bottom=0,top=250)
 ax4.set_xlabel('Experiment frame number')
 ax4.set_ylabel('Millimeters')
 ax4.set_title('Average 3D estimation error per frame') 
 #ax4.set_xticklabels(labels, rotation=45, rotation_mode="anchor")
 ax4.legend()
#-----------------------------------------------------------------------------------------------------------------------







#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
addedPixelNoise=0.0
procrustesScale=False
drawPlot=0
everyNFrames=0
doProcrustes=1
h36MDataFileUsesMocapNETLabels=0

scalingFileTarget='scalingTraget.conf'
generateScalingRulesRun=0
#python3 compareHuman36AndMyCSVFiles.py --h36m /home/ammar/Documents/Programming/DNNTracker/DNNTracker/dataset/Directions-1/human36_2DGroundTruthAsMockOpenPoseOutput/54138969/h36.csv --mocapnet /home/ammar/Documents/Programming/DNNTracker/DNNTracker/dataset/Directions-1/human36_2DGroundTruthAsMockOpenPoseOutput/54138969/3d_compare.csv --out /home/ammar/Documents/Programming/DNNTracker/DNNTracker/dataset/Directions-1/human36_2DGroundTruthAsMockOpenPoseOutput/54138969/stats.csv


#------------------------------------------------------------------------------------------------ 
if (len(sys.argv)>1):
   #print('Argument List:', str(sys.argv))
   for i in range(0, len(sys.argv)):
       if (sys.argv[i]=="--generateScalingRules"): 
          generateScalingRulesRun=1
          scalingFileTarget=sys.argv[i+1] 
       if (sys.argv[i]=="--noprocrustes"): 
          doProcrustes=0
       if (sys.argv[i]=="--every"): 
          everyNFrames=int(sys.argv[i+1])
       if (sys.argv[i]=="--draw"): 
          drawPlot=1
       if (sys.argv[i]=="--scale"): 
          procrustesScale=True 
       if (sys.argv[i]=="--h36m"):
          print("\n Human 3.6M CSV at ",sys.argv[i+1])
          h36mCSVPath=sys.argv[i+1] 
       if (sys.argv[i]=="--mocapnet"):
          print("\n Mocapnet CSV at ",sys.argv[i+1]);
          mocapNETCSVPath=sys.argv[i+1] 
       if (sys.argv[i]=="--out"):
          print("\n Output CSV at ",sys.argv[i+1])
          outCSVPath=sys.argv[i+1]
       if (sys.argv[i]=="--cc"):
          print("\n Output CC CSV at ",sys.argv[i+1])
          outCCCSVPath=sys.argv[i+1]
       if (sys.argv[i]=="--info"):
          print("\n Infos Set ",sys.argv[i+1])
          subject=sys.argv[i+1]
          action=sys.argv[i+2]
          subaction=sys.argv[i+3]
          camera=sys.argv[i+4]
          actionLabel=sys.argv[i+5]
          addedPixelNoise=float(sys.argv[i+6])
          print("Subject ",subject," Action ",actionLabel,"Camera ",camera)
#------------------------------------------------------------------------------------------------ 
   
 

if (drawPlot):
   # === Plot and animate === 
   fig = plt.figure()
   fig.set_size_inches(19.2, 10.8, forward=True) 
 
   ax = fig.add_subplot(2, 2, 1, projection='3d') 
   ax2 = fig.add_subplot(2, 2, 2) 
   ax3 = fig.add_subplot(2, 2, 3) 
   ax4 = fig.add_subplot(2, 2, 4) 
   fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95) 
 
   ax.set_xlabel('X Axis')
   ax.set_ylabel('Y Axis')
   ax.set_zlabel('Z Axis') 
   ax.view_init(90, 90) 






#----------------------------------------------------------------------------------
#                              h36M Dataset Loading
#----------------------------------------------------------------------------------
#h36MFile = readCSVFileFloatBody(h36mCSVPath)

print("Human36M header:")
print(h36MFile['header'])

#h36MFileSelected=justKeepTheseCSVColumns(h36MFile,WHATTOKEEPFROMOPENPOSEJOINTS)

print("h36MFileSelected All Possible joints where:")
print(WHATTOKEEPFROMOPENPOSEJOINTS)
print("h36MFileSelected Selected header:")
print(h36MFileSelected['header'])


numberOfFrames=int(len(h36MFileSelected['body']))
numberOfJoints=int(len(h36MFileSelected['header'])/3)
print("Number of Frames: ",numberOfFrames," Number Of Joints: ",numberOfJoints)

#if (generateScalingRulesRun):
#    generateRulesForScalingMnet(scalingFileTarget,h36MFileSelected['body'],numberOfFrames,numberOfJoints)
#    sys.exit(0)
 
#----------------------------------------------------------------------------------


#----------------------------------------------------------------------------------
#                              MocapNET Dataset Loading
#----------------------------------------------------------------------------------
#MocapNETFile = readCSVFileFloatBody(mocapNETCSVPath)

print("MocapNET header:")
print(MocapNETFile['header']) 


#MocapNETFileSelected=justKeepTheseCSVColumns(MocapNETFile,WHATTOKEEPFROMMOCAPNETJOINTS)
print("MocapNET All Possible joints where:")
print(WHATTOKEEPFROMMOCAPNETJOINTS)
print("MocapNET Selected header:")
print(MocapNETFileSelected['header'])
#----------------------------------------------------------------------------------

if ( len(h36MFile['body']) != len(MocapNETFile['body'])  ):
    print("Mismatch of body sizes for h36M file / MocapNET file ") 
    sys.exit(0)
print("\n\n\n\n\n\n")



#print(MocapNETFileSelected['body'])



for jointID in range(0,numberOfJoints):
    print("Joint %u => %s (parent %s) "%(jointID,JOINT_LABELS[jointID],JOINT_LABELS[JOINT_PARENTS[jointID]]));

sanityCheckLimbDimensions(h36MFileSelected['body'],MocapNETFileSelected['body'],numberOfFrames,numberOfJoints)

#-----------------------
alljointDistances=list()
#-----------------------
jointDistance=list()
for jointID in range(0,numberOfJoints):
    jointDistance.append(list())
#-----------------------

totalSamples=0
totalError=0

averageErrorDistances=list()
averageMotionEstimationDistancesBetweenFrames=list()
executeComputations=1

for frameID in range(0,numberOfFrames):
 if ( (everyNFrames==0) or (frameID%everyNFrames==0) ):
   executeComputations=1
 else:
   executeComputations=0
 
 if (executeComputations):
   currentDistances=list()
   pointList3DOur=list() 
   pointList3DH36=list()
   xlineStart=list()
   xlineEnd=list()
   ylineStart=list()
   ylineEnd=list()
   zlineStart=list()
   zlineEnd=list()
   xs=list()
   ys=list()
   zs=list()
   sys.stdout.write('.')
   sys.stdout.flush()
 
   hipJoint=8
   #Calculate hip offsets to align two skeletons using translation only
   hipDistX=h36MFileSelected['body'][frameID][hipJoint*3+0]-10*MocapNETFileSelected['body'][frameID][hipJoint*3+0]
   hipDistY=h36MFileSelected['body'][frameID][hipJoint*3+1]+10*MocapNETFileSelected['body'][frameID][hipJoint*3+1]
   hipDistZ=h36MFileSelected['body'][frameID][hipJoint*3+2]+10*MocapNETFileSelected['body'][frameID][hipJoint*3+2]
 
   #FOR JOINT START -----------------------------------
   for jointID in range(0,numberOfJoints):
      #------------------------------------------------------------------ 
      jointParentID=JOINT_PARENTS[jointID] 
      #------------------------------------------------------------------ 
      #ignore all points that don't have a name 
      xM= 10*MocapNETFileSelected['body'][frameID][jointID*3+0]+hipDistX
      yM=-10*MocapNETFileSelected['body'][frameID][jointID*3+1]+hipDistY
      zM=-10*MocapNETFileSelected['body'][frameID][jointID*3+2]+hipDistZ 
      #------------------------------------------------------------------ 
      xH=h36MFileSelected['body'][frameID][jointID*3+0]
      yH=h36MFileSelected['body'][frameID][jointID*3+1]
      zH=h36MFileSelected['body'][frameID][jointID*3+2] 
      #------------------------------------------------------------------ 
      pointList3DOur.append([xM,yM,zM])
      pointList3DH36.append([xH,yH,zH])
      #------------------------------------------------------------------ 
   #FOR JOINT END -----------------------------------


   h36pcl =  np.asarray(pointList3DH36,dtype=np.float32)
   ourpcl =  np.asarray(pointList3DOur,dtype=np.float32)
   
  
   distance = pointListsReturnAvgDistance(ourpcl,h36pcl)
   disparity = distance   
   

   if (doProcrustes):
       d, Z, T, b, c = compute_similarity_transform(h36pcl,ourpcl,compute_optimal_scale=procrustesScale)
       disparity=np.sqrt(d)  #d: squared error after transformation

       #Our point cloud is brought to the same translation and rotation as h36 point cloud
       ourpcl = (b*ourpcl.dot(T))+c
       disparity = pointListsReturnAvgDistance(ourpcl,h36pcl)
   
   if (frameID==0):
     ourPreviousPCL=ourpcl
   
   totalMotionFromPreviousFrame=0
   for jointID in range(0,numberOfJoints): 
       #--------------------------------------- 
       jointParentID=JOINT_PARENTS[jointID]
       jX=ourpcl[jointID][0]
       jY=ourpcl[jointID][1]
       jZ=ourpcl[jointID][2]
       pX=ourPreviousPCL[jointID][0]
       pY=ourPreviousPCL[jointID][1]
       pZ=ourPreviousPCL[jointID][2]
       totalMotionFromPreviousFrame=totalMotionFromPreviousFrame + get3DDistance(jX,jY,jZ,pX,pY,pZ) 
   averageMotionEstimationDistancesBetweenFrames.append(totalMotionFromPreviousFrame/numberOfJoints)
   ourPreviousPCL=ourpcl

   averageErrorDistances.append(disparity)

   #Calculate per joint distances --------
   for jointID in range(0,numberOfJoints):
       jointParentID=JOINT_PARENTS[jointID]
       #--------------------------------
       #--------------------------------
       #  Retrieve the MocapNET point 
       #--------------------------------
       #--------------------------------
       xM=ourpcl[jointID][0] 
       yM=ourpcl[jointID][1]
       zM=ourpcl[jointID][2]
       #--------------------
       xParentM=ourpcl[jointParentID][0]
       yParentM=ourpcl[jointParentID][1]
       zParentM=ourpcl[jointParentID][2]
       xlineStart.append(xM) 
       ylineStart.append(yM) 
       zlineStart.append(zM) 
       xlineEnd.append(xParentM) 
       ylineEnd.append(yParentM) 
       zlineEnd.append(zParentM)
       #--------------------------------
       #--------------------------------
       #--------------------------------

       #--------------------------------
       #--------------------------------
       #    Retrieve the H36M point 
       #--------------------------------
       #--------------------------------
       xH=h36pcl[jointID][0] 
       yH=h36pcl[jointID][1]
       zH=h36pcl[jointID][2]
       #--------------------
       xParentH=h36pcl[jointParentID][0]
       yParentH=h36pcl[jointParentID][1]
       zParentH=h36pcl[jointParentID][2]
       #--------------------
       xlineStart.append(xH) 
       ylineStart.append(yH) 
       zlineStart.append(zH) 
       xlineEnd.append(xParentH) 
       ylineEnd.append(yParentH) 
       zlineEnd.append(zParentH)
       #--------------------------------
       #--------------------------------
       #--------------------------------


       # Get the euclidean 3D distance of the two points..
       dist= get3DDistance(xM,yM,zM,xH,yH,zH)
       currentDistances.append(dist)

       alljointDistances.append(dist)
       jointDistance[jointID].append(dist)
   #--------------------------------------
 
   if (drawPlot): 
      plt.cla() 
      ax.cla()
      ax2.cla()
      ax3.cla()
      ax4.cla()

      #print Skeletons and their connected lines.. 
      for i in range(0,len(xlineStart)): 
          ax.plot([xlineStart[i],xlineEnd[i]],[ylineStart[i],ylineEnd[i]],zs=[zlineStart[i],zlineEnd[i]])
      xs, ys, zs = pointListReturnXYZListForScatterPlot(ourpcl)
      ax.scatter(xs, ys, zs)
      xs, ys, zs = pointListReturnXYZListForScatterPlot(h36pcl)
      ax.scatter(xs, ys, zs) 

      #Secondary plots for limb lengths etc.. 
      drawLimbDimensions(h36pcl,ourpcl,numberOfJoints,ax2) 
      #drawLimbError(h36pcl,ourpcl,numberOfJoints,ax3)
      drawFrameError(averageErrorDistances,averageMotionEstimationDistancesBetweenFrames,ax4)
 
      #------------------------- 
      ax.text2D(0.05, 0.95, "Frame %u/%u - Increment %u - Procrustes Average Error %0.2f mm"%(frameID,numberOfFrames,everyNFrames,disparity) , transform=ax.transAxes)
      #ax.text2D(0.05, 0.05, "RHand %0.2f mm / LHand %0.2f mm / RFoot %0.2f mm / LFoot %0.2f mm "%(currentDistances[4],currentDistances[7],currentDistances[11],currentDistances[14]) , transform=ax.transAxes)
      #------------------------- 
      ax.set_xlim(auto=False,left=-600,right=300)
      ax.set_ylim(auto=False,bottom=-1600,top=200)
      ax.set_zlim(auto=False,bottom=2000,top=6000)
      ax.set_xlabel('X Axis')
      ax.set_ylabel('Y Axis')
      ax.set_zlabel('Z Axis') 
      #------------------------- 
      plt.show(block=False) 
      plt.savefig('p%05u.png'%frameID)
      #fig.canvas.draw() 
      plt.pause(0.0001)
      #------------------------- 

   totalError+=disparity
   totalSamples+=1

   print("Procrustes(%0.2f)/Reg.(%0.2f) "%(disparity,distance), end=" ")
 #-------------------------------------------------------------------
#writeCSVFileResults(outCSVPath,1,alljointDistances,jointDistance,hipJoint,numberOfJoints,subject,action,subaction,camera,actionLabel,addedPixelNoise)
#writeRAWResultsForGNUplot("%s-gnuplot.raw" % outCSVPath,1,alljointDistances)


writeHeader=0
if not os.path.exists(outCCCSVPath):
   writeHeader=1
#writeCSVFileResults(outCCCSVPath,writeHeader,alljointDistances,jointDistance,hipJoint,numberOfJoints,subject,action,subaction,camera,actionLabel,addedPixelNoise)
#writeRAWResultsForGNUplot("%s-gnuplot.raw" % outCCCSVPath,writeHeader,alljointDistances)

print("\nAverage Error for ",totalSamples," samples is ",totalError/totalSamples)
median=np.median(alljointDistances)
mean=np.mean(alljointDistances)
average=np.average(alljointDistances)
std=np.std(alljointDistances)
var=np.var(alljointDistances)
print("Avg is ",average," Std is ",std," Var is ",var)

if (drawPlot): 
  os.system("ffmpeg -framerate 25 -i p%05d.png  -s 1920x1080  -y -r 30 -pix_fmt yuv420p -threads 8  lastcomp.mp4 && rm ./p*.png") #
sys.exit(0)
