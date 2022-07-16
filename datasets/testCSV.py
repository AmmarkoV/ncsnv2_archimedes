#Simple CSV loading sample - Ammar Qammaz

import numpy as np 
import os
import csv
import gc
import time
import array
import sys

"""
Bytes to KB/MB/GB/TB converter
"""
def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    step_unit = 1000.0 #1024 bad the size

    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < step_unit:
            return "%3.1f %s" % (num, x)
        num /= step_unit

"""
Check if a file exists
"""
def checkIfFileExists(filename):
    return os.path.isfile(filename) 

"""
Count the number of lines by parsing the file inside python
"""
def getNumberOfLines(filename):
    print("Counting number of lines in file ",filename)
    with open(filename) as f:
        return sum(1 for line in f) 


"""
    typically we work with data[NUMBER_OF_SAMPLES,NUMBER_OF_COLUMNS] 
    calling splitNumpyArray(data,4,4,0) means get back columns 4,5,6,7 so it will return a data[NUMBER_OF_SAMPLES,4] 
"""
def splitNumpyArray(data,column,columnsToTake,useHalfFloats=False):
    #-------------------------------------
    dtypeSelected=np.dtype(np.float32)
    if (useHalfFloats):
       dtypeSelected=np.dtype(np.float16)
    #-------------------------------------
    numberOfSamples=len(data)
    npOutput = np.full([numberOfSamples,columnsToTake],fill_value=0,dtype=dtypeSelected,order='C')
    
    for outCol in range(0,columnsToTake):
           for num in range(0,numberOfSamples):
                    npOutput[num,outCol]=float(data[num,column+outCol])
    #-------------------------------------
    return npOutput;



#---------------------------------------------------------------------------------------------------------------------------------------------
def readCSVFile(filename,memPercentage=1.0,useHalfFloats=False):
    print("CSV file :",filename,"..\n")

    if (not checkIfFileExists(filename)):
          print( bcolors.WARNING + "Input file "+filename+" does not exist, cannot read ground truth.." +  bcolors.ENDC)
          print("Current Directory was "+os.getcwd())
          sys.exit(0) 
    start = time.time()
     
    dtypeSelected=np.dtype(np.float32)  
    dtypeSelectedByteSize=int(dtypeSelected.itemsize) 
    if (useHalfFloats):
       dtypeSelected=np.dtype(np.float16) 
       dtypeSelectedByteSize=int(dtypeSelected.itemsize)

    progress=0.0
    useSquared=1
    sampleNumber=0
    receivedHeader=0
    inputNumberOfColumns=0 
    outputNumberOfColumns=0 

    inputLabels=list() 

    #-------------------------------------------------------------------------------------------------
    numberOfSamplesInput=getNumberOfLines(filename)-2 
    print(" Input file has ",numberOfSamplesInput," training samples\n")
    #-------------------------------------------------------------------------------------------------


    numberOfSamples = numberOfSamplesInput
    numberOfSamplesLimit=int(numberOfSamples*memPercentage)
    #------------------------------------------------------------------------------------------------- 
    if (memPercentage==0.0):
        print("readGroundTruthFile was asked to occupy 0 memory so this probably means we just want one record")    
        numberOfSamplesLimit=2
    if (memPercentage>1.0):
        print("Memory Limit will be interpreted as a raw value..")     
        numberOfSamplesLimit=int(memPercentage)
    #-------------------------------------------------------------------------------------------------

 
    thisInput = array.array('f')  
    #---------------------------------  

    fi = open(filename, "r") 
    readerIn  = csv.reader( fi , delimiter =',', skipinitialspace=True) 
    for rowIn in readerIn: 
        #------------------------------------------------------
        if (not receivedHeader): #use header to get labels
           #------------------------------------------------------
           inputNumberOfColumns=len(rowIn)
           inputLabels = list(rowIn[i] for i in range(0,inputNumberOfColumns) )
           print("Number of Input elements : ",len(inputLabels))
           #------------------------------------------------------

           if (memPercentage==0):
               print("Will only return labels\n")
               return {'labels':inputLabels};
           #i=0
           #print("class Input(Enum):")
           #for label in inputLabels:
           #   print("     ",label," = ",i," #",int(i/3))
           #   print("     ",label,"=",int(i/3))
           #   i=i+1

           #---------------------------------  
           #         Allocate Lists 
           #---------------------------------
           for i in range(inputNumberOfColumns):
               thisInput.append(0.0)
           #---------------------------------  


           #---------------------------------  
           #      Allocate Numpy Arrays 
           #---------------------------------  
           inputSize=0
           startCompressed=0
           inputSize=inputNumberOfColumns
           startCompressed=inputNumberOfColumns  
           
           npInputBytesize=0+numberOfSamplesLimit * inputSize * dtypeSelectedByteSize
           print(" Input file on disk has a shape of [",numberOfSamples,",",inputSize,"]")  
           print(" Input we will read has a shape of [",numberOfSamplesLimit,",",inputSize,"]")  
           print(" Input will occupy ",convert_bytes(npInputBytesize)," of RAM\n")  
           npInput = np.full([numberOfSamplesLimit,inputSize],fill_value=0,dtype=dtypeSelected,order='C')
           #----------------------------------------------------------------------------------------------------------
           receivedHeader=1 
           #sys.exit(0)
        else:
           #-------------------------------------------
           #  First convert our string INPUT to floats   
           #-------------------------------------------
           for i in range(inputNumberOfColumns):
                  thisInput[i]=float(rowIn[i]) 
           #------------------------------------------- 
           for num in range(0,inputNumberOfColumns):
                  npInput[sampleNumber,num]=float(thisInput[num]); 
           #-------------------------------------------
           sampleNumber=sampleNumber+1

        if (numberOfSamples>0):
           progress=sampleNumber/numberOfSamplesLimit 

        if (sampleNumber%1000==0) :
           progressString = "%0.2f"%float(100*progress)  
           print("\rReading from disk (",sampleNumber,") - ",progressString," %      \r", end="", flush=True)  

        if (numberOfSamplesLimit<=sampleNumber):
           print("\rStopping reading file to obey memory limit given by parameter --mem ",memPercentage,"\n")
           break
    #-------------------------------------------
    fi.close()
    del readerIn
    gc.collect()


    print("\n read, Samples: ",sampleNumber,", was expecting ",numberOfSamples," samples\n") 
    print(npInput.shape)

    totalNumberOfBytes=npInput.nbytes;
    totalNumberOfGigaBytes=totalNumberOfBytes/1073741824; 
    print("GPU Size Occupied by data = ",totalNumberOfGigaBytes," GB \n")

    end = time.time()
    print("Time elapsed : ",(end-start)/60," mins")
    #---------------------------------------------------------------------
    return {'label':inputLabels, 'body':npInput };








if __name__ == '__main__':

    os.system('./getCMUBVH.sh') 

    data2D = readCSVFile("2d_body_all.csv")
    print("DATA 2D : ",data2D['label'])

    #Get stats for all 2D joints      
    for columnToProcess in range(0,len(data2D['label'])):
        outputName = data2D['label'][columnToProcess]
        print("We will process 2D column number ",columnToProcess," aka ",outputName)
        singleColumn = splitNumpyArray(data2D['body'],columnToProcess,1)
        print("We have ",len(singleColumn)," samples" )
        minimum      = np.min(singleColumn)
        maximum      = np.max(singleColumn)
        median       = np.median(singleColumn)
        mean         = np.mean(singleColumn)
        std          = np.std(singleColumn)
        var          = np.var(singleColumn)
        title_string = " %s : Min/Max=%0.2f/%0.2f,Median=%0.2f,Mean=%0.2f,Std=%0.2f,Var=%0.2f" % (outputName,minimum,maximum,median,mean,std,var)
        print("Statistics of Output ",title_string)


    data3D = readCSVFile("3d_body_all.csv")
    print("DATA 3D : ",data3D['label'])

    #Get stats for all 3D joints      
    for columnToProcess in range(0,len(data2D['label'])):
        outputName = data3D['label'][columnToProcess]
        print("We will process 3D column number ",columnToProcess," aka ",outputName)
        singleColumn = splitNumpyArray(data3D['body'],columnToProcess,1)
        print("We have ",len(singleColumn)," samples" )
        minimum      = np.min(singleColumn)
        maximum      = np.max(singleColumn)
        median       = np.median(singleColumn)
        mean         = np.mean(singleColumn)
        std          = np.std(singleColumn)
        var          = np.var(singleColumn)
        title_string = " %s : Min/Max=%0.2f/%0.2f,Median=%0.2f,Mean=%0.2f,Std=%0.2f,Var=%0.2f" % (outputName,minimum,maximum,median,mean,std,var)
        print("Statistics of Output ",title_string)





