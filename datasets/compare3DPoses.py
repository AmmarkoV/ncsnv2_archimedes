#!/usr/bin/env python3
import h5py
import numpy as np
import csv
import os
import sys
from os import listdir
from os.path import isdir, join
import xml.etree.ElementTree as ET

from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes

from align2DPoints import pointListReturnXYZListForScatterPlot,pointListsReturnAvgDistance,compute_similarity_transform
#from procrustes import procrustes_aligned
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d, Axes3D  
print("Using matplotlib:",matplotlib.__version__)
 
 


#These are not used but kept for reference 
#-----------------------------------------------
ORIGINAL_OPENPOSE_NAMES = ['']*26
ORIGINAL_OPENPOSE_NAMES[0]='Nose'        #0
ORIGINAL_OPENPOSE_NAMES[1]='Neck'        #1
ORIGINAL_OPENPOSE_NAMES[2]='RShoulder'   #2
ORIGINAL_OPENPOSE_NAMES[3]='RElbow'      #3
ORIGINAL_OPENPOSE_NAMES[4]='RWrist'      #4
ORIGINAL_OPENPOSE_NAMES[5]='LShoulder'   #5
ORIGINAL_OPENPOSE_NAMES[6]='LElbow'      #6
ORIGINAL_OPENPOSE_NAMES[7]='LWrist'      #7
ORIGINAL_OPENPOSE_NAMES[8]='MidHip'      #8
ORIGINAL_OPENPOSE_NAMES[9]='RHip'        #9
ORIGINAL_OPENPOSE_NAMES[10]='RKnee'      #10
ORIGINAL_OPENPOSE_NAMES[11]='RAnkle'     #11
ORIGINAL_OPENPOSE_NAMES[12]='LHip'       #12
ORIGINAL_OPENPOSE_NAMES[13]='LKnee'      #13
ORIGINAL_OPENPOSE_NAMES[14]='LAnkle'     #14
ORIGINAL_OPENPOSE_NAMES[15]='REye'       #15
ORIGINAL_OPENPOSE_NAMES[16]='LEye'       #16
ORIGINAL_OPENPOSE_NAMES[17]='REar'       #17
ORIGINAL_OPENPOSE_NAMES[18]='LEar'       #18
ORIGINAL_OPENPOSE_NAMES[19]='LBigToe'    #19
ORIGINAL_OPENPOSE_NAMES[20]='LSmallToe'  #20
ORIGINAL_OPENPOSE_NAMES[21]='LHeel'      #21
ORIGINAL_OPENPOSE_NAMES[22]='RBigToe'    #22
ORIGINAL_OPENPOSE_NAMES[23]='RSmallToe'  #23
ORIGINAL_OPENPOSE_NAMES[24]='RHeel'      #24
ORIGINAL_OPENPOSE_NAMES[25]='Background' #25 <- this does not exist
 


#These are not used but kept for reference 
#-----------------------------------------------
ORIGINAL_OPENPOSE_PARENTS = [0]*26
ORIGINAL_OPENPOSE_PARENTS[0]  = 1
ORIGINAL_OPENPOSE_PARENTS[1]  = 8
ORIGINAL_OPENPOSE_PARENTS[2]  = 1
ORIGINAL_OPENPOSE_PARENTS[3]  = 2
ORIGINAL_OPENPOSE_PARENTS[4]  = 3
ORIGINAL_OPENPOSE_PARENTS[5]  = 1
ORIGINAL_OPENPOSE_PARENTS[6]  = 5
ORIGINAL_OPENPOSE_PARENTS[7]  = 6
ORIGINAL_OPENPOSE_PARENTS[8]  = 8
ORIGINAL_OPENPOSE_PARENTS[9]  = 8
ORIGINAL_OPENPOSE_PARENTS[10]  = 9
ORIGINAL_OPENPOSE_PARENTS[11]  = 10
ORIGINAL_OPENPOSE_PARENTS[12] = 8
ORIGINAL_OPENPOSE_PARENTS[13] = 12
ORIGINAL_OPENPOSE_PARENTS[14] = 13
ORIGINAL_OPENPOSE_PARENTS[15] = 0
ORIGINAL_OPENPOSE_PARENTS[16] = 0
ORIGINAL_OPENPOSE_PARENTS[17] = 15
ORIGINAL_OPENPOSE_PARENTS[18] = 16
ORIGINAL_OPENPOSE_PARENTS[19] = 14
ORIGINAL_OPENPOSE_PARENTS[20] = 14
ORIGINAL_OPENPOSE_PARENTS[21] = 14
ORIGINAL_OPENPOSE_PARENTS[22] = 11
ORIGINAL_OPENPOSE_PARENTS[23] = 11
ORIGINAL_OPENPOSE_PARENTS[24] = 11
ORIGINAL_OPENPOSE_PARENTS[25] = 8 


#These are not used but kept for reference 
#-----------------------------------------------
MNET_DIMENSIONS = [0.0]*26
MNET_DIMENSIONS[0]  = 0.0 #Statistics of distance of Joint  Nose  to Joint  Neck
MNET_DIMENSIONS[1]  = 544.3934 #Statistics of distance of Joint  Neck  to Joint  MidHip
MNET_DIMENSIONS[2]  = 125.14453 / 1.1430199 #Statistics of distance of Joint  RShoulder  to Joint  Neck
MNET_DIMENSIONS[3]  = 282.30347 #Statistics of distance of Joint  RElbow  to Joint  RShoulder
MNET_DIMENSIONS[4]  = 210.58955 #Statistics of distance of Joint  RWrist  to Joint  RElbow
MNET_DIMENSIONS[5]  = 125.14453 / 1.1430199 #Statistics of distance of Joint  LShoulder  to Joint  Neck
MNET_DIMENSIONS[6]  = 282.30347 #Statistics of distance of Joint  LElbow  to Joint  LShoulder
MNET_DIMENSIONS[7]  = 210.5896 #Statistics of distance of Joint  LWrist  to Joint  LElbow
MNET_DIMENSIONS[8]  = 0.0 #Statistics of distance of Joint  MidHip  to Joint  MidHip
MNET_DIMENSIONS[9]  = 92.19512 #Statistics of distance of Joint  RHip  to Joint  MidHip
MNET_DIMENSIONS[10] = 368.27164 #Statistics of distance of Joint  RKnee  to Joint  RHip
MNET_DIMENSIONS[11] = 454.06 #Statistics of distance of Joint  RHeel  to Joint  RKnee
MNET_DIMENSIONS[12] = 92.19512 #Statistics of distance of Joint  LHip  to Joint  MidHip
MNET_DIMENSIONS[13] = 368.27164 #Statistics of distance of Joint  LKnee  to Joint  LHip
MNET_DIMENSIONS[14] = 454.06 #Statistics of distance of Joint  LHeel  to Joint  LKnee
MNET_DIMENSIONS[15] = 0.0
MNET_DIMENSIONS[16] = 0.0
MNET_DIMENSIONS[17] = 0.0
MNET_DIMENSIONS[18] = 0.0
MNET_DIMENSIONS[19] = 0.0
MNET_DIMENSIONS[20] = 0.0
MNET_DIMENSIONS[21] = 0.0
MNET_DIMENSIONS[22] = 0.0
MNET_DIMENSIONS[23] = 0.0
MNET_DIMENSIONS[24] = 0.0
MNET_DIMENSIONS[25] = 0.0
  
 
  

#These are our labels they have to match WHATTOKEEPFROMOPENPOSEJOINTS
#---------------------------------------------------------------------
JOINT_LABELS = ['']*26
JOINT_LABELS[0]='Nose'        #0
JOINT_LABELS[1]='Neck'        #1
JOINT_LABELS[2]='RShoulder'   #2
JOINT_LABELS[3]='RElbow'      #3
JOINT_LABELS[4]='RWrist'      #4
JOINT_LABELS[5]='LShoulder'   #5
JOINT_LABELS[6]='LElbow'      #6
JOINT_LABELS[7]='LWrist'      #7
JOINT_LABELS[8]='MidHip'      #8
JOINT_LABELS[9]='RHip'        #9
JOINT_LABELS[10]='RKnee'      #10
JOINT_LABELS[11]='RHeel'      #11
JOINT_LABELS[12]='LHip'       #12
JOINT_LABELS[13]='LKnee'      #13
JOINT_LABELS[14]='LHeel'      #14
JOINT_LABELS[15]='RBigToe'    #15
JOINT_LABELS[16]='LBigToe'    #16
JOINT_LABELS[17]='REar'       #17
JOINT_LABELS[18]='LEar'       #18
JOINT_LABELS[19]='LBigToe'    #19
JOINT_LABELS[20]='LSmallToe'  #20
JOINT_LABELS[21]='LHeel'      #21
JOINT_LABELS[22]='RBigToe'    #22
JOINT_LABELS[23]='RSmallToe'  #23
JOINT_LABELS[24]='RHeel'      #24
JOINT_LABELS[25]='Background' #25 <- this does not exist
                  

#These are our parents they have to match WHATTOKEEPFROMOPENPOSEJOINTS
#---------------------------------------------------------------------
JOINT_PARENTS = [0]*26
JOINT_PARENTS[0]  = 1 #Parent of Nose is Neck
JOINT_PARENTS[1]  = 8 #Parent of Neck is MidHip
JOINT_PARENTS[2]  = 1 #Parent of RShoulder is Neck
JOINT_PARENTS[3]  = 2 #Parent of RElbow is RShoulder
JOINT_PARENTS[4]  = 3 
JOINT_PARENTS[5]  = 1
JOINT_PARENTS[6]  = 5
JOINT_PARENTS[7]  = 6
JOINT_PARENTS[8]  = 8
JOINT_PARENTS[9]  = 8
JOINT_PARENTS[10]  = 9
JOINT_PARENTS[11]  = 10
JOINT_PARENTS[12] = 8
JOINT_PARENTS[13] = 12
JOINT_PARENTS[14] = 13
JOINT_PARENTS[15] = 11
JOINT_PARENTS[16] = 14
JOINT_PARENTS[17] = 15
JOINT_PARENTS[18] = 16
JOINT_PARENTS[19] = 14
JOINT_PARENTS[20] = 14
JOINT_PARENTS[21] = 14
JOINT_PARENTS[22] = 11
JOINT_PARENTS[23] = 11
JOINT_PARENTS[24] = 11
JOINT_PARENTS[25] = 8 


NUMBER_OF_JOINT_COORDINATES_USED_IN_COMPARISON = 45 #45/51
WHATTOKEEPFROMOPENPOSEJOINTS = ['']*NUMBER_OF_JOINT_COORDINATES_USED_IN_COMPARISON 
#-----------------------------------------            
WHATTOKEEPFROMOPENPOSEJOINTS[0]='3DX_Neck'#'3DX_Nose'    
WHATTOKEEPFROMOPENPOSEJOINTS[1]='3DY_Neck'#'3DY_Nose'    
WHATTOKEEPFROMOPENPOSEJOINTS[2]='3DZ_Neck'#'3DZ_Nose'  
#-----------------------------------------        
WHATTOKEEPFROMOPENPOSEJOINTS[3]='3DX_Neck' 
WHATTOKEEPFROMOPENPOSEJOINTS[4]='3DY_Neck'    
WHATTOKEEPFROMOPENPOSEJOINTS[5]='3DZ_Neck'   
#-----------------------------------------        
WHATTOKEEPFROMOPENPOSEJOINTS[6]='3DX_RShoulder'        
WHATTOKEEPFROMOPENPOSEJOINTS[7]='3DY_RShoulder'        
WHATTOKEEPFROMOPENPOSEJOINTS[8]='3DZ_RShoulder'     
#-----------------------------------------        
WHATTOKEEPFROMOPENPOSEJOINTS[9]='3DX_RElbow'        
WHATTOKEEPFROMOPENPOSEJOINTS[10]='3DY_RElbow'        
WHATTOKEEPFROMOPENPOSEJOINTS[11]='3DZ_RElbow'        
#-----------------------------------------        
WHATTOKEEPFROMOPENPOSEJOINTS[12]='3DX_RWrist'        
WHATTOKEEPFROMOPENPOSEJOINTS[13]='3DY_RWrist'        
WHATTOKEEPFROMOPENPOSEJOINTS[14]='3DZ_RWrist'        
#-----------------------------------------        
WHATTOKEEPFROMOPENPOSEJOINTS[15]='3DX_LShoulder'        
WHATTOKEEPFROMOPENPOSEJOINTS[16]='3DY_LShoulder'        
WHATTOKEEPFROMOPENPOSEJOINTS[17]='3DZ_LShoulder'        
#-----------------------------------------        
WHATTOKEEPFROMOPENPOSEJOINTS[18]='3DX_LElbow'        
WHATTOKEEPFROMOPENPOSEJOINTS[19]='3DY_LElbow'        
WHATTOKEEPFROMOPENPOSEJOINTS[20]='3DZ_LElbow'        
#-----------------------------------------        
WHATTOKEEPFROMOPENPOSEJOINTS[21]='3DX_LWrist'        
WHATTOKEEPFROMOPENPOSEJOINTS[22]='3DY_LWrist'        
WHATTOKEEPFROMOPENPOSEJOINTS[23]='3DZ_LWrist'        
#-----------------------------------------        
WHATTOKEEPFROMOPENPOSEJOINTS[24]='3DX_MidHip'        
WHATTOKEEPFROMOPENPOSEJOINTS[25]='3DY_MidHip'        
WHATTOKEEPFROMOPENPOSEJOINTS[26]='3DZ_MidHip'        
#-----------------------------------------        
WHATTOKEEPFROMOPENPOSEJOINTS[27]='3DX_RHip'        
WHATTOKEEPFROMOPENPOSEJOINTS[28]='3DY_RHip'        
WHATTOKEEPFROMOPENPOSEJOINTS[29]='3DZ_RHip'        
#-----------------------------------------        
WHATTOKEEPFROMOPENPOSEJOINTS[30]='3DX_RKnee'        
WHATTOKEEPFROMOPENPOSEJOINTS[31]='3DY_RKnee'        
WHATTOKEEPFROMOPENPOSEJOINTS[32]='3DZ_RKnee'        
#-----------------------------------------        
WHATTOKEEPFROMOPENPOSEJOINTS[33]='3DX_RHeel' #3DX_RAnkle        
WHATTOKEEPFROMOPENPOSEJOINTS[34]='3DY_RHeel' #3DY_RAnkle            
WHATTOKEEPFROMOPENPOSEJOINTS[35]='3DZ_RHeel' #3DZ_RAnkle              
#-----------------------------------------        
WHATTOKEEPFROMOPENPOSEJOINTS[36]='3DX_LHip'        
WHATTOKEEPFROMOPENPOSEJOINTS[37]='3DY_LHip'        
WHATTOKEEPFROMOPENPOSEJOINTS[38]='3DZ_LHip'        
#-----------------------------------------        
WHATTOKEEPFROMOPENPOSEJOINTS[39]='3DX_LKnee'        
WHATTOKEEPFROMOPENPOSEJOINTS[40]='3DY_LKnee'        
WHATTOKEEPFROMOPENPOSEJOINTS[41]='3DZ_LKnee'        
#-----------------------------------------        
WHATTOKEEPFROMOPENPOSEJOINTS[42]='3DX_LHeel' #3DX_LAnkle         
WHATTOKEEPFROMOPENPOSEJOINTS[43]='3DY_LHeel' #3DY_LAnkle            
WHATTOKEEPFROMOPENPOSEJOINTS[44]='3DZ_LHeel' #3DZ_LAnkle     

#set to 45 to stop here
if (NUMBER_OF_JOINT_COORDINATES_USED_IN_COMPARISON>=51):
  WHATTOKEEPFROMOPENPOSEJOINTS[45]='3DX_RBigToe'         
  WHATTOKEEPFROMOPENPOSEJOINTS[46]='3DY_RBigToe'         
  WHATTOKEEPFROMOPENPOSEJOINTS[47]='3DZ_RBigToe'         
  #-----------------------------------------        
  WHATTOKEEPFROMOPENPOSEJOINTS[48]='3DX_LBigToe'         
  WHATTOKEEPFROMOPENPOSEJOINTS[49]='3DY_LBigToe'         
  WHATTOKEEPFROMOPENPOSEJOINTS[50]='3DZ_LBigToe'      
  #set to 51 to stop here   

  
#set to 45 to stop here
if (NUMBER_OF_JOINT_COORDINATES_USED_IN_COMPARISON>=57):
  WHATTOKEEPFROMOPENPOSEJOINTS[51]='3DX_REye'         
  WHATTOKEEPFROMOPENPOSEJOINTS[52]='3DY_REye'         
  WHATTOKEEPFROMOPENPOSEJOINTS[53]='3DZ_REye'         
  #-----------------------------------------        
  WHATTOKEEPFROMOPENPOSEJOINTS[54]='3DX_LEye'         
  WHATTOKEEPFROMOPENPOSEJOINTS[55]='3DY_LEye'         
  WHATTOKEEPFROMOPENPOSEJOINTS[56]='3DZ_LEye'      
  #set to 51 to stop here   



WHATTOKEEPFROMMOCAPNETJOINTS = ['']*NUMBER_OF_JOINT_COORDINATES_USED_IN_COMPARISON 
#-----------------------------------------         
WHATTOKEEPFROMMOCAPNETJOINTS[0]='3DX_Neck'  #'3DX_oris06'#'3DX_EndSite_oris05' # '3DX_Head'        
WHATTOKEEPFROMMOCAPNETJOINTS[1]='3DY_Neck'  #'3DY_oris06'#'3DY_EndSite_oris05'  # '3DY_Head'        
WHATTOKEEPFROMMOCAPNETJOINTS[2]='3DZ_Neck'  #'3DZ_oris06'#'3DZ_EndSite_oris05'  # '3DZ_Head'  
#-----------------------------------------        
WHATTOKEEPFROMMOCAPNETJOINTS[3]='3DX_Neck'        
WHATTOKEEPFROMMOCAPNETJOINTS[4]='3DY_Neck'        
WHATTOKEEPFROMMOCAPNETJOINTS[5]='3DZ_Neck'        
#-----------------------------------------        
WHATTOKEEPFROMMOCAPNETJOINTS[6]='3DX_RShoulder'        
WHATTOKEEPFROMMOCAPNETJOINTS[7]='3DY_RShoulder'        
WHATTOKEEPFROMMOCAPNETJOINTS[8]='3DZ_RShoulder'        
#-----------------------------------------        
WHATTOKEEPFROMMOCAPNETJOINTS[9]='3DX_RElbow'        
WHATTOKEEPFROMMOCAPNETJOINTS[10]='3DY_RElbow'        
WHATTOKEEPFROMMOCAPNETJOINTS[11]='3DZ_RElbow'        
#-----------------------------------------        
WHATTOKEEPFROMMOCAPNETJOINTS[12]='3DX_RHand'        
WHATTOKEEPFROMMOCAPNETJOINTS[13]='3DY_RHand'        
WHATTOKEEPFROMMOCAPNETJOINTS[14]='3DZ_RHand'        
#-----------------------------------------        
WHATTOKEEPFROMMOCAPNETJOINTS[15]='3DX_LShoulder'        
WHATTOKEEPFROMMOCAPNETJOINTS[16]='3DY_LShoulder'        
WHATTOKEEPFROMMOCAPNETJOINTS[17]='3DZ_LShoulder'        
#-----------------------------------------        
WHATTOKEEPFROMMOCAPNETJOINTS[18]='3DX_LElbow'        
WHATTOKEEPFROMMOCAPNETJOINTS[19]='3DY_LElbow'        
WHATTOKEEPFROMMOCAPNETJOINTS[20]='3DZ_LElbow'        
#-----------------------------------------        
WHATTOKEEPFROMMOCAPNETJOINTS[21]='3DX_LHand'        
WHATTOKEEPFROMMOCAPNETJOINTS[22]='3DY_LHand'        
WHATTOKEEPFROMMOCAPNETJOINTS[23]='3DZ_LHand'        
#-----------------------------------------        
WHATTOKEEPFROMMOCAPNETJOINTS[24]='3DX_Hip'        
WHATTOKEEPFROMMOCAPNETJOINTS[25]='3DY_Hip'        
WHATTOKEEPFROMMOCAPNETJOINTS[26]='3DZ_Hip'        
#-----------------------------------------        
WHATTOKEEPFROMMOCAPNETJOINTS[27]='3DX_RHip'        
WHATTOKEEPFROMMOCAPNETJOINTS[28]='3DY_RHip'        
WHATTOKEEPFROMMOCAPNETJOINTS[29]='3DZ_RHip'        
#-----------------------------------------        
WHATTOKEEPFROMMOCAPNETJOINTS[30]='3DX_RKnee'        
WHATTOKEEPFROMMOCAPNETJOINTS[31]='3DY_RKnee'        
WHATTOKEEPFROMMOCAPNETJOINTS[32]='3DZ_RKnee'        
#-----------------------------------------        
WHATTOKEEPFROMMOCAPNETJOINTS[33]='3DX_RFoot'        
WHATTOKEEPFROMMOCAPNETJOINTS[34]='3DY_RFoot'        
WHATTOKEEPFROMMOCAPNETJOINTS[35]='3DZ_RFoot'        
#-----------------------------------------        
WHATTOKEEPFROMMOCAPNETJOINTS[36]='3DX_LHip'        
WHATTOKEEPFROMMOCAPNETJOINTS[37]='3DY_LHip'        
WHATTOKEEPFROMMOCAPNETJOINTS[38]='3DZ_LHip'        
#-----------------------------------------        
WHATTOKEEPFROMMOCAPNETJOINTS[39]='3DX_LKnee'        
WHATTOKEEPFROMMOCAPNETJOINTS[40]='3DY_LKnee'        
WHATTOKEEPFROMMOCAPNETJOINTS[41]='3DZ_LKnee'        
#-----------------------------------------        
WHATTOKEEPFROMMOCAPNETJOINTS[42]='3DX_LFoot'         
WHATTOKEEPFROMMOCAPNETJOINTS[43]='3DY_LFoot'         
WHATTOKEEPFROMMOCAPNETJOINTS[44]='3DZ_LFoot'         
#set to 45 to stop here
if (NUMBER_OF_JOINT_COORDINATES_USED_IN_COMPARISON>=51):
  WHATTOKEEPFROMMOCAPNETJOINTS[45]='3DX_EndSite_toe1-2.r'         
  WHATTOKEEPFROMMOCAPNETJOINTS[46]='3DY_EndSite_toe1-2.r'         
  WHATTOKEEPFROMMOCAPNETJOINTS[47]='3DZ_EndSite_toe1-2.r'         
  #-----------------------------------------        
  WHATTOKEEPFROMMOCAPNETJOINTS[48]='3DX_EndSite_toe1-2.l'         
  WHATTOKEEPFROMMOCAPNETJOINTS[49]='3DY_EndSite_toe1-2.l'         
  WHATTOKEEPFROMMOCAPNETJOINTS[50]='3DZ_EndSite_toe1-2.l'      
  #set to 51 to stop here        
if (NUMBER_OF_JOINT_COORDINATES_USED_IN_COMPARISON>=57):
  WHATTOKEEPFROMMOCAPNETJOINTS[51]='3DX_EndSite_eye.r'         
  WHATTOKEEPFROMMOCAPNETJOINTS[52]='3DY_EndSite_eye.r'         
  WHATTOKEEPFROMMOCAPNETJOINTS[53]='3DZ_EndSite_eye.r'         
  #-----------------------------------------        
  WHATTOKEEPFROMMOCAPNETJOINTS[54]='3DX_EndSite_eye.l'         
  WHATTOKEEPFROMMOCAPNETJOINTS[55]='3DY_EndSite_eye.l'         
  WHATTOKEEPFROMMOCAPNETJOINTS[56]='3DZ_EndSite_eye.l'      
  #set to 51 to stop here        

def get3DDistance(jX,jY,jZ,pX,pY,pZ):
 return np.sqrt( ((jX-pX)*(jX-pX)) + ((jY-pY)*(jY-pY)) + ((jZ-pZ)*(jZ-pZ)) )

#-----------------------------------------------------------------------------------------------------------------------
def writeCSVFileResults(outputFile,addHeader,globalJointDistances,perJointDistance,hipJoint,numberOfJoints,subject,action,subaction,camera,actionLabel,addedPixelNoise):
  #Write header----------------
  if (addHeader):
      file = open(outputFile,"w")  
      file.write("Subject,")
      file.write("Action,")
      file.write("ActionLabel,")
      file.write("Subaction,")
      file.write("Camera,")
      file.write("Noise,")
      file.write("Global_Median,")
      #file.write("Global_Mean,")
      file.write("Global_Average,")
      file.write("Global_Std,")
      file.write("Global_Var,")
      for jointID in range(0,numberOfJoints):
        if (jointID!=hipJoint):
            file.write("%s_Median,"%JOINT_LABELS[jointID])
            #file.write("%s_Mean,"%JOINT_LABELS[jointID])
            file.write("%s_Average,"%JOINT_LABELS[jointID])
            file.write("%s_Std,"%JOINT_LABELS[jointID])
            file.write("%s_Var"%JOINT_LABELS[jointID])
            if (jointID!=numberOfJoints-1):
             file.write(",")
            else:   
             file.write("\n")
  else:
      file = open(outputFile,"a")  
  #---------------------------- 

  median=np.median(globalJointDistances)
  mean=np.mean(globalJointDistances)
  average=np.average(globalJointDistances)
  std=np.std(globalJointDistances)
  var=np.var(globalJointDistances)
  print("\nGlobal Median:",median," Average:",average," Std:",std,"Var:",var)    #file.write(subject)
  file.write(subject)
  file.write(",")
  file.write(action)
  file.write(",")
  file.write(actionLabel)
  file.write(",")
  file.write(subaction)
  file.write(",")
  file.write(camera)  
  file.write(",")
  file.write(str(addedPixelNoise))  
  file.write(",")
  file.write(str(median))
  file.write(",")
  #file.write(str(mean))
  #file.write(",")
  file.write(str(average))
  file.write(",")
  file.write(str(std))
  file.write(",")
  file.write(str(var))
  file.write(",")
  
  for jointID in range(0,numberOfJoints):
    if (jointID!=hipJoint):
        median=np.median(jointDistance[jointID])
        mean=np.mean(jointDistance[jointID])
        average=np.average(jointDistance[jointID])
        std=np.std(jointDistance[jointID])
        var=np.var(jointDistance[jointID])
        print("Joint ",JOINT_LABELS[jointID]," Median:",median," Mean:",mean," Average:",average," Std:",std,"Var:",var)
        file.write(str(median))
        file.write(",")
        #file.write(str(mean))
        #file.write(",")
        file.write(str(average))
        file.write(",")
        file.write(str(std))
        file.write(",")
        file.write(str(var))   
        if (jointID!=numberOfJoints-1):
         file.write(",")
        else:   
         file.write("\n")

  file.close()
#--------------------------------------------



#-----------------------------------------------------------------------------------------------------------------------
def writeRAWResultsForGNUplot(outputFile,addHeader,globalJointDistances):
    if (addHeader):
       fileHandler = open(outputFile, "w")
    else:
       fileHandler = open(outputFile, "a")
    for measurement in globalJointDistances:
        fileHandler.write(str(measurement))
        fileHandler.write("\n")
    fileHandler.close()
         


#-----------------------------------------------------------------------------------------------------------------------
def readCSVFileFloatBody(filenameInput):
 #--------------------------------------------
 sampleNumber=0
 fileHandler = open(filenameInput, "r")
 csvReader = csv.reader(fileHandler,delimiter =',',skipinitialspace=True)
 bodyList=list()  
 inputLabels=list()
 #--------------------------------------------
 for rowIn in csvReader: 
        #------------------------------------------------------
        if (sampleNumber==0): #use header to get labels
           #------------------------------------------------------
           inputNumberOfColumns=len(rowIn)
           inputLabels = list(rowIn[i].lower() for i in range(0,inputNumberOfColumns) ) 
           #------------------------------------------------------
        else: 
           inputNumberOfColumns=len(rowIn)
           inputValues = list(float(rowIn[i]) for i in range(0,inputNumberOfColumns) )  
           bodyList.append(inputValues)
        sampleNumber=sampleNumber+1
 fileHandler.close()
 return {'header':inputLabels, 'body':bodyList};
#-----------------------------------------------------------------------------------------------------------------------




#-----------------------------------------------------------------------------------------------------------------------
def justKeepTheseCSVColumns(fullCSVList,whatToKeep):
 #--------------------------------------------
 newBodyList=list()  
 newInputLabels=list()
 listOfColumnsToKeep=list()
 #--------------------------------------------
 for labelToKeep in whatToKeep:
        i=0
        for headerEntry in fullCSVList['header']:
            #Convert all labels to lowercase to make sure we don't lose something..
            if labelToKeep.lower()==headerEntry.lower(): 
               print("Found ",labelToKeep," in column ",i) 
               listOfColumnsToKeep.append(i)
            i=i+1

 if (len(whatToKeep)!=len(listOfColumnsToKeep)):
   print("Failure while trying to select list items..")
   print("Columns Requested (%u)"%len(whatToKeep))
   print(whatToKeep)
   print("Columns Found (%u) "%len(listOfColumnsToKeep))
   print(listOfColumnsToKeep)
   sys.exit(0)

 #print("listOfColumnsToKeep:")
 #print(listOfColumnsToKeep)
 #--------------------------------------------
 #------------------HEADER--------------------
 #--------------------------------------------
 for i in range(0,len(listOfColumnsToKeep)):
        val=listOfColumnsToKeep[i]
        newInputLabels.append(fullCSVList['header'][val])
 #--------------------------------------------
 #------------------BODY----------------------
 #--------------------------------------------
 for row in range(0,len(fullCSVList['body'])):
     thisNewRow=list()
     for i in range(0,len(listOfColumnsToKeep)):
         val=listOfColumnsToKeep[i]
         thisNewRow.append(fullCSVList['body'][row][val])
         #print("newInputLabels ",fullCSVList['header'][val]," in val ",val) 
     newBodyList.append(thisNewRow)
 #--------------------------------------------
 return {'header':newInputLabels, 'body':newBodyList};
#-----------------------------------------------------------------------------------------------------------------------



def generateRulesForScalingMnet(outputFile,h36mPoints,numberOfFrames,numberOfJoints):

 h36m_joint_distances = list() 
 h36m_averages = list() 

 for jointID in range(0,numberOfJoints):
     h36m_joint_distances.append(list()) 

 for frameID in range(0,numberOfFrames):
   pointList3DH36=list() 
   for jointID in range(0,numberOfJoints):
      xH36M=h36mPoints[frameID][jointID*3+0]
      yH36M=h36mPoints[frameID][jointID*3+1]
      zH36M=h36mPoints[frameID][jointID*3+2] 
      pointList3DH36.append([xH36M,yH36M,zH36M]) 
     
   h36pcl =  np.asarray(pointList3DH36,dtype=np.float32) 

   for jointID in range(0,numberOfJoints):
       h36m_distances = list() 
       #--------------------------------------- 
       jointParentID=JOINT_PARENTS[jointID]
       #--------------------------------------- 
       jX=h36pcl[jointID][0]
       jY=h36pcl[jointID][1]
       jZ=h36pcl[jointID][2]
       pX=h36pcl[jointParentID][0]
       pY=h36pcl[jointParentID][1]
       pZ=h36pcl[jointParentID][2]
       distances=get3DDistance(jX,jY,jZ,pX,pY,pZ) 
       h36m_joint_distances[jointID].append(distances) 
 

 rulesFile = open(outputFile, 'w')


 for jointID in range(0,numberOfJoints):
     #--------------------------------------- 
     jointParentID=JOINT_PARENTS[jointID]
     #---------------------------------------------------
     minimumH36M=np.min(h36m_joint_distances[jointID])
     maximumH36M=np.max(h36m_joint_distances[jointID])
     medianH36M=np.median(h36m_joint_distances[jointID])
     averageH36M=np.average(h36m_joint_distances[jointID])
     h36m_averages.append(averageH36M)
     #--------------------------------------------------- 
     print("Statistics of distance of Joint ",JOINT_LABELS[jointID]," to Joint ",JOINT_LABELS[jointParentID])
     print(" H36M Min:",minimumH36M," Max:",maximumH36M," Avg:",averageH36M)
     averageM = MNET_DIMENSIONS[jointID]
     print(" MocapNET  Avg:",averageM)
     if (averageM!=0.0):
           print(" Scale value by ",averageH36M/averageM)
   
 rulesFile.write('torsoLength=%f\n'% (h36m_averages[1]/MNET_DIMENSIONS[1]))
 rulesFile.write('shoulderToElbowLength=%f\n'% (h36m_averages[3]/MNET_DIMENSIONS[3]))
 rulesFile.write('elbowToHandLength=%f\n'% (h36m_averages[4]/MNET_DIMENSIONS[4]))
 rulesFile.write('chestWidth=%f\n'% (h36m_averages[2]/MNET_DIMENSIONS[2]))
 rulesFile.write('waistWidth=%f\n'% (h36m_averages[9]/MNET_DIMENSIONS[9]))
 rulesFile.write('hipToKneeLength=%f\n'% (h36m_averages[10]/MNET_DIMENSIONS[10]))
 rulesFile.write('kneeToFootLength=%f\n'% (h36m_averages[11]/MNET_DIMENSIONS[11]))
 rulesFile.write('kneeToFootLength=%f\n'% (h36m_averages[11]/MNET_DIMENSIONS[11]))

 rulesFile.close()
 
# sys.exit(0) 
#-----------------------------------------------------------------------------------------------------------------------




#-----------------------------------------------------------------------------------------------------------------------
def sanityCheckLimbDimensions(h36mPoints,ourPoints,numberOfFrames,numberOfJoints):
 h36m_joint_distances = list()
 mnet_joint_distances = list()

 for jointID in range(0,numberOfJoints):
     h36m_joint_distances.append(list())
     mnet_joint_distances.append(list())

 for frameID in range(0,numberOfFrames):
   pointList3DH36=list()
   pointList3DOur=list() 
   for jointID in range(0,numberOfJoints):
      xH36M=h36mPoints[frameID][jointID*3+0]
      yH36M=h36mPoints[frameID][jointID*3+1]
      zH36M=h36mPoints[frameID][jointID*3+2] 
      pointList3DH36.append([xH36M,yH36M,zH36M])
    
      xM=10*ourPoints[frameID][jointID*3+0]
      yM=-10*ourPoints[frameID][jointID*3+1]
      zM=-10*ourPoints[frameID][jointID*3+2] 
      pointList3DOur.append([xM,yM,zM])
     
     
   h36pcl =  np.asarray(pointList3DH36,dtype=np.float32)
   ourpcl =  np.asarray(pointList3DOur,dtype=np.float32)

   for jointID in range(0,numberOfJoints):
       h36m_distances = list()
       mnet_distances = list()
       #--------------------------------------- 
       jointParentID=JOINT_PARENTS[jointID]
       #--------------------------------------- 
       jX=h36pcl[jointID][0]
       jY=h36pcl[jointID][1]
       jZ=h36pcl[jointID][2]
       pX=h36pcl[jointParentID][0]
       pY=h36pcl[jointParentID][1]
       pZ=h36pcl[jointParentID][2]
       distances=get3DDistance(jX,jY,jZ,pX,pY,pZ) 
       h36m_joint_distances[jointID].append(distances)
       #--------------------------------------- 
       jX=ourpcl[jointID][0]
       jY=ourpcl[jointID][1]
       jZ=ourpcl[jointID][2]
       pX=ourpcl[jointParentID][0]
       pY=ourpcl[jointParentID][1]
       pZ=ourpcl[jointParentID][2]
       distances=get3DDistance(jX,jY,jZ,pX,pY,pZ) 
       mnet_joint_distances[jointID].append(distances)
       #--------------------------------------- 

 for jointID in range(0,numberOfJoints):
     #--------------------------------------- 
     jointParentID=JOINT_PARENTS[jointID]
     #---------------------------------------------------
     minimumH36M=np.min(h36m_joint_distances[jointID])
     maximumH36M=np.max(h36m_joint_distances[jointID])
     medianH36M=np.median(h36m_joint_distances[jointID])
     averageH36M=np.average(h36m_joint_distances[jointID])
     #---------------------------------------------------
     minimumM=np.min(mnet_joint_distances[jointID])
     maximumM=np.max(mnet_joint_distances[jointID])
     medianM=np.median(mnet_joint_distances[jointID])
     averageM=np.average(mnet_joint_distances[jointID])
     #---------------------------------------------------
     print("Statistics of distance of Joint ",JOINT_LABELS[jointID]," to Joint ",JOINT_LABELS[jointParentID])
     print(" H36M Min:",minimumH36M," Max:",maximumH36M," Avg:",averageH36M)
     print(" MocapNET Min:",minimumM," Max:",maximumM," Avg:",averageM)
     if (averageM!=0.0):
           print(" Scale value by ",averageH36M/averageM)
   

# sys.exit(0) 
#-----------------------------------------------------------------------------------------------------------------------




#-----------------------------------------------------------------------------------------------------------------------
def drawLimbDimensions(h36mPoints,ourPoints,numberOfJoints,ax2):
 labels = list()
 h36m_distances = list()
 mnet_distances = list()
 for jointID in range(0,numberOfJoints):
       #--------------------------------------- 
       labels.append(JOINT_LABELS[jointID])
       #--------------------------------------- 
       jointParentID=JOINT_PARENTS[jointID]
       jX=h36mPoints[jointID][0]
       jY=h36mPoints[jointID][1]
       jZ=h36mPoints[jointID][2]
       pX=h36mPoints[jointParentID][0]
       pY=h36mPoints[jointParentID][1]
       pZ=h36mPoints[jointParentID][2]
       distances=get3DDistance(jX,jY,jZ,pX,pY,pZ) 
       h36m_distances.append(distances)
       #--------------------------------------- 
       jX=ourPoints[jointID][0]
       jY=ourPoints[jointID][1]
       jZ=ourPoints[jointID][2]
       pX=ourPoints[jointParentID][0]
       pY=ourPoints[jointParentID][1]
       pZ=ourPoints[jointParentID][2]
       distances=get3DDistance(jX,jY,jZ,pX,pY,pZ) 
       mnet_distances.append(distances)
       #--------------------------------------- 

 #------------------------------------------------- 
 x = np.arange(len(labels))  # the label locations
 width = 0.35  # the width of the bars

 rects1 = ax2.bar(x - width/2, h36m_distances, width, label='H36M')
 rects2 = ax2.bar(x + width/2, mnet_distances, width, label='Our Method')

  # Add some text for labels, title and custom x-axis tick labels, etc.
 ax2.set_ylim(auto=False,bottom=0,top=800)
 ax2.set_ylabel('Limb dimensions in millimeters')
 ax2.set_title('Comparison of H36M limb dimensions')
 ax2.set_xticks(x)
 ax2.set_xticklabels(labels, rotation=45, rotation_mode="anchor")
 ax2.legend()

 #autolabel(rects1)
 #autolabel(rects2)
#-----------------------------------------------------------------------------------------------------------------------




#-----------------------------------------------------------------------------------------------------------------------
def drawLimbError(h36mPoints,ourPoints,numberOfJoints,ax3):
 labels = list()
 error_distances = list() 
 for jointID in range(0,numberOfJoints):
       #--------------------------------------- 
       labels.append(JOINT_LABELS[jointID])
       #--------------------------------------- 
       jointParentID=JOINT_PARENTS[jointID]
       jX=h36mPoints[jointID][0]
       jY=h36mPoints[jointID][1]
       jZ=h36mPoints[jointID][2]
       pX=ourPoints[jointID][0]
       pY=ourPoints[jointID][1]
       pZ=ourPoints[jointID][2]
       distances=get3DDistance(jX,jY,jZ,pX,pY,pZ) 
       error_distances.append(distances)
       #---------------------------------------  

 #------------------------------------------------- 
 x = np.arange(len(labels))  # the label locations
 width = 0.35  # the width of the bars

 rects1 = ax3.bar(x - width/2, error_distances, width, label='Error in millimeters') 

  # Add some text for labels, title and custom x-axis tick labels, etc.
 ax3.set_ylim(auto=False,bottom=0,top=250)
 ax3.set_ylabel('Error in millimeters')
 ax3.set_title('Comparison of 3D estimation error')
 ax3.set_xticks(x)
 ax3.set_xticklabels(labels, rotation=45, rotation_mode="anchor")
 ax3.legend()
#-----------------------------------------------------------------------------------------------------------------------



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
h36MFile = readCSVFileFloatBody(h36mCSVPath)

print("Human36M header:")
print(h36MFile['header'])

h36MFileSelected=justKeepTheseCSVColumns(h36MFile,WHATTOKEEPFROMOPENPOSEJOINTS)

print("h36MFileSelected All Possible joints where:")
print(WHATTOKEEPFROMOPENPOSEJOINTS)
print("h36MFileSelected Selected header:")
print(h36MFileSelected['header'])


numberOfFrames=int(len(h36MFileSelected['body']))
numberOfJoints=int(len(h36MFileSelected['header'])/3)
print("Number of Frames: ",numberOfFrames," Number Of Joints: ",numberOfJoints)

if (generateScalingRulesRun):
    generateRulesForScalingMnet(scalingFileTarget,h36MFileSelected['body'],numberOfFrames,numberOfJoints)
    sys.exit(0)
 
#----------------------------------------------------------------------------------


#----------------------------------------------------------------------------------
#                              MocapNET Dataset Loading
#----------------------------------------------------------------------------------
MocapNETFile = readCSVFileFloatBody(mocapNETCSVPath)

print("MocapNET header:")
print(MocapNETFile['header']) 


MocapNETFileSelected=justKeepTheseCSVColumns(MocapNETFile,WHATTOKEEPFROMMOCAPNETJOINTS)
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
      drawLimbError(h36pcl,ourpcl,numberOfJoints,ax3)
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
writeCSVFileResults(outCSVPath,1,alljointDistances,jointDistance,hipJoint,numberOfJoints,subject,action,subaction,camera,actionLabel,addedPixelNoise)
writeRAWResultsForGNUplot("%s-gnuplot.raw" % outCSVPath,1,alljointDistances)


writeHeader=0
if not os.path.exists(outCCCSVPath):
   writeHeader=1
writeCSVFileResults(outCCCSVPath,writeHeader,alljointDistances,jointDistance,hipJoint,numberOfJoints,subject,action,subaction,camera,actionLabel,addedPixelNoise)
writeRAWResultsForGNUplot("%s-gnuplot.raw" % outCCCSVPath,writeHeader,alljointDistances)

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
