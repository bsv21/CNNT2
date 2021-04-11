# Note - this is a script to prepare the DESS data to be input into the cartilage mask generating network.
# This script is very similar to the "prepareOAIimageData" script and eventually these should be combined.
# We refer to "prepareOAIimageData" for more clarification of the code functionality.

import os
import numpy as np
import pydicom
from scipy.ndimage import zoom
from scipy.interpolate import RegularGridInterpolator
import tensorflow as tf
import png
from PIL import Image
from pathlib import Path
import scipy.io as sio
import h5py

# Type in path to dicoms - see comments to "prepareOAIimageData"
basePath =
# Type in path to write output - see comments to "prepareOAIimageData"
outputDir =
patientDirList = []
patientCount = 0


def scanInfo(scanData,ds,sliceLocs,sliceCount,echoTimes,echoTimeCount,voxpos,cubeShape,dx,dy,minPositionPatient,maxPositionPatient,minmaxSliceLocation):
              
  sliceIndexMatrix = np.array([ np.repeat(np.arange(cubeShape[1]),cubeShape[2]),
                                np.tile(np.arange(cubeShape[2]),cubeShape[1]),
                                np.zeros(cubeShape[1]*cubeShape[2]),
                                np.ones(cubeShape[1]*cubeShape[2]) ])
      
  foundSliceLoc = np.where(sliceLocs == ds.SliceLocation)
  sliceCount_new = sliceCount
  if not any(map(len, foundSliceLoc)):
    A = np.array([ [ds.ImageOrientationPatient[3]*dx, ds.ImageOrientationPatient[0]*dy, 0, ds.ImagePositionPatient[0]],
                   [ds.ImageOrientationPatient[4]*dx, ds.ImageOrientationPatient[1]*dy, 0, ds.ImagePositionPatient[1]],
                   [ds.ImageOrientationPatient[5]*dx, ds.ImageOrientationPatient[2]*dy, 0, ds.ImagePositionPatient[2]],
                   [0, 0, 0, 1] ])
    P = np.matmul(A,sliceIndexMatrix)
    voxpos[sliceCount,:,:,0] = P[0,:].reshape((cubeShape[1],cubeShape[2]))
    voxpos[sliceCount,:,:,1] = P[1,:].reshape((cubeShape[1],cubeShape[2]))
    voxpos[sliceCount,:,:,2] = P[2,:].reshape((cubeShape[1],cubeShape[2]))

    if ds.SliceLocation < minmaxSliceLocation[0]:
      minmaxSliceLocation[0] = ds.SliceLocation
      minPositionPatient[:] = ds.ImagePositionPatient[:]
    if ds.SliceLocation > minmaxSliceLocation[1]:
      minmaxSliceLocation[1] = ds.SliceLocation
      maxPositionPatient[:] = ds.ImagePositionPatient[:]

    sliceLocs[sliceCount] = ds.SliceLocation
    sliceCount_new += 1
  
  sliceLocIndexArray = np.where(sliceLocs==ds.SliceLocation)
  sliceLocIndex = sliceLocIndexArray[0][0]

  echoTimeCount_new = echoTimeCount
  foundTE = np.where(echoTimes == ds.EchoTime)
  if not any(map(len, foundTE)):
    # This TE is not in the array of found TEs (have not processed a slice with this TE before)
    echoTimes[echoTimeCount] = ds.EchoTime
    echoTimeCount_new += 1
    
  echoTimeIndexArray = np.where(echoTimes==ds.EchoTime)
  echoTimeIndex = echoTimeIndexArray[0][0]
    
  scanData[echoTimeIndex,sliceLocIndex,:,:] = ds.pixel_array; 
  
  return sliceCount_new, echoTimeCount_new


def interPolateToT2grid(scanData, cubeShape, minPosPatient, maxPosPatient, orientationPatient, dx, dy, PT2_3D):
  zVec = [minPosPatient[0] - maxPosPatient[0], minPosPatient[1] - maxPosPatient[1], minPosPatient[2] - maxPosPatient[2]]
  # Compute matrix to transform 3D dataset from T2 index frame to scanner physical coordinates frame.
  A = np.array([ [orientationPatient[3]*dx, orientationPatient[0]*dy, zVec[0]/(1-cubeShape[0]), minPosPatient[0]],
                 [orientationPatient[4]*dx, orientationPatient[1]*dy, zVec[1]/(1-cubeShape[0]), minPosPatient[1]],
                 [orientationPatient[5]*dx, orientationPatient[2]*dy, zVec[2]/(1-cubeShape[0]), minPosPatient[2]],
                 [0, 0, 0, 1] ])
  C_T2 = np.linalg.solve(A, PT2_3D)
      
  interp = RegularGridInterpolator((np.arange(cubeShape[1]), np.arange(cubeShape[2]), np.arange(cubeShape[0])),
                                    np.transpose(scanData,(1,2,0)), bounds_error=False, fill_value=0)
  scanDataInT2grid = interp(C_T2[:3,:].T)
  return scanDataInT2grid




########################

# Make a list of all the patient directories in the base directory  
for (patientDirpath, patientDirnames, patientFilenames) in os.walk(basePath):
  patientDirList.extend(patientDirnames)
  break
patientDirList.sort()
# Loop through the patient directories
for patientPath in patientDirList:
  generatedImages = 0
  
  dateDirList = []
  dateCount = 0
  
  # Make a list of all the exam dates in this patient directory
  for (dateDirpath, dateDirnames, dateFilenames) in os.walk(basePath + patientPath + "/"):
    dateDirList.extend(dateDirnames)
    break
  dateDirList.sort()

  print(basePath + patientPath + "/")
  print(dateDirList)
  
  
  sequenceDirList = []
  seqCount = 0
  
  for datePath in dateDirList:
    seqCount += 1
    print(datePath + "\n")
    for (sequenceDirpath, sequenceDirnames, sequenceFilenames) in os.walk(basePath + patientPath + "/" + datePath + "/"):
    	sequenceDirList.extend(sequenceDirnames)
    	break
    sequenceDirList.sort()
    print(sequenceDirList)
    
    COR_TSE_found = 0
    cortseMinPositionPatient = [0, 0, 0]
    cortseMaxPositionPatient = [100, 100, 100]
    minmaxCorTSEsliceLocation = [10000, -10000]
    corTSEechoTimes = np.zeros(1)
    corTSEechoTimeCount = 0
    
    SAG_DESS_found = 0
    dessMinPositionPatient = [0, 0, 0]
    dessMaxPositionPatient = [100, 100, 100]
    minmaxDESSsliceLocation = [10000, -10000]
    sagDESSechoTimes = np.zeros(1)
    sagDESSechoTimeCount = 0
    
    COR_FLASH_found = 0
    corflashMinPositionPatient = [0, 0, 0]
    corflashMaxPositionPatient = [100, 100, 100]
    minmaxCorFLASHsliceLocation = [10000, -10000]
    corFLASHechoTimes = np.zeros(1)
    corFLASHechoTimeCount = 0
    
    SAG_TSE_found = 0
    sagtseMinPositionPatient = [0, 0, 0]
    sagtseMaxPositionPatient = [100, 100, 100]
    minmaxSagTSEsliceLocation = [10000, -10000]
    sagTSEechoTimes = np.zeros(1)
    sagTSEechoTimeCount = 0
    
    T2_found = 0
    T2MinPositionPatient = [0, 0, 0]
    T2MaxPositionPatient = [100, 100, 100]
    minmaxT2sliceLocation = [10000, -10000]
    T2echoTimes = np.zeros(7)
    T2echoTimeCount = 0
    
    for seq in sequenceDirList:
    
        sliceCount = 0
    
        fileList = []
        sequencepath = sequenceDirpath + seq + "/"
        print(sequencepath)
        for (dirpath, dirnames, filenames) in os.walk(sequencepath):
            fileList.extend(filenames)
            break
        fileList.sort()
        nFiles = int(len(fileList))
        sliceLocs = np.zeros(nFiles)
    
    
        for ff in fileList:
          dicomPath = sequencepath + ff
          ds = pydicom.dcmread(dicomPath);
          
          ###############################
          # COR FLASH scans
          if ("FLASH" in ds.SeriesDescription) and ("RIGHT" in ds.SeriesDescription):
            if (COR_FLASH_found == 0):
              corFLASHcubeShape = (nFiles, (ds.pixel_array).shape[0], (ds.pixel_array).shape[1])    # (z,x,y)
              corflash_dx =  float(ds.PixelSpacing[0])
              corflash_dy =  float(ds.PixelSpacing[1])
              corFLASHvoxpos = np.zeros(corFLASHcubeShape + (3,))
              COR_FLASH_scans = np.zeros((1,) + corFLASHcubeShape)
              # Save the orientation vectors for later interpolating voxel values onto T2 grid.
              corflashOrientationPatient = ds.ImageOrientationPatient
              COR_FLASH_sliceLocs = np.zeros(nFiles)
              COR_FLASH_found = 1
 
            (sliceCount_new, echoTimeCount_new) = scanInfo(COR_FLASH_scans, ds, COR_FLASH_sliceLocs, sliceCount, corFLASHechoTimes, corFLASHechoTimeCount, corFLASHvoxpos, corFLASHcubeShape, corflash_dx, corflash_dy, corflashMinPositionPatient, corflashMaxPositionPatient, minmaxCorFLASHsliceLocation)
            sliceCount = sliceCount_new
            corFLASHechoTimeCount = echoTimeCount_new
          
          
          ###############################
          # SAG DESS scans
          if ("DESS" in ds.SeriesDescription) and ("RIGHT" in ds.SeriesDescription):
            if (SAG_DESS_found == 0):
              sagDESScubeShape = (nFiles, (ds.pixel_array).shape[0], (ds.pixel_array).shape[1])    # (z,x,y)
              dess_dx =  float(ds.PixelSpacing[0])
              dess_dy =  float(ds.PixelSpacing[1])
              sagDESSvoxpos = np.zeros(sagDESScubeShape + (3,))
              SAG_DESS_scans = np.zeros((1,) + sagDESScubeShape)
              # Save the orientation vectors for later interpolating voxel values onto T2 grid.
              dessOrientationPatient = ds.ImageOrientationPatient
              SAG_DESS_sliceLocs = np.zeros(nFiles)
              SAG_DESS_found = 1
 
            (sliceCount_new, echoTimeCount_new) = scanInfo(SAG_DESS_scans, ds, SAG_DESS_sliceLocs, sliceCount, sagDESSechoTimes, sagDESSechoTimeCount, sagDESSvoxpos, sagDESScubeShape, dess_dx, dess_dy, dessMinPositionPatient, dessMaxPositionPatient, minmaxDESSsliceLocation)
            sliceCount = sliceCount_new
            sagDESSechoTimeCount = echoTimeCount_new
    
    
          ###############################
          # COR TSE scans
          if ("COR" in ds.SeriesDescription) and ("TSE" in ds.SeriesDescription) and ("RIGHT" in ds.SeriesDescription):
            if (COR_TSE_found == 0):
              corTSEcubeShape = (nFiles, (ds.pixel_array).shape[0], (ds.pixel_array).shape[1])    # (z,x,y)
              cortse_dx =  float(ds.PixelSpacing[0])
              cortse_dy =  float(ds.PixelSpacing[1])
              corTSEvoxpos = np.zeros(corTSEcubeShape + (3,))
              COR_TSE_scans = np.zeros((1,) + corTSEcubeShape)
              # Save the orientation vectors for later interpolating voxel values onto T2 grid.
              cortseOrientationPatient = ds.ImageOrientationPatient
              COR_TSE_sliceLocs = np.zeros(nFiles)
              COR_TSE_found = 1

 
            (sliceCount_new, echoTimeCount_new) = scanInfo(COR_TSE_scans, ds, COR_TSE_sliceLocs, sliceCount, corTSEechoTimes, corTSEechoTimeCount, corTSEvoxpos, corTSEcubeShape, cortse_dx, cortse_dy, cortseMinPositionPatient, cortseMaxPositionPatient, minmaxCorTSEsliceLocation)
            sliceCount = sliceCount_new
            corTSEechoTimeCount = echoTimeCount_new

    
          ###############################
          # SAG TSE scans
          if ("SAG" in ds.SeriesDescription) and ("TSE" in ds.SeriesDescription) and ("RIGHT" in ds.SeriesDescription):
            if (SAG_TSE_found == 0):
              sagTSEcubeShape = (nFiles, (ds.pixel_array).shape[0], (ds.pixel_array).shape[1])    # (z,x,y)
              sagtse_dx =  float(ds.PixelSpacing[0])
              sagtse_dy =  float(ds.PixelSpacing[1])
              sagTSEvoxpos = np.zeros(sagTSEcubeShape + (3,))
              SAG_TSE_scans = np.zeros((1,) + sagTSEcubeShape)
              # Save the orientation vectors for later interpolating voxel values onto T2 grid.
              sagtseOrientationPatient = ds.ImageOrientationPatient
              SAG_TSE_sliceLocs = np.zeros(nFiles)
              SAG_TSE_found = 1
 
            (sliceCount_new, echoTimeCount_new) = scanInfo(SAG_TSE_scans, ds, SAG_TSE_sliceLocs, sliceCount, sagTSEechoTimes, sagTSEechoTimeCount, sagTSEvoxpos, sagTSEcubeShape, sagtse_dx, sagtse_dy, sagtseMinPositionPatient, sagtseMaxPositionPatient, minmaxSagTSEsliceLocation)
            sliceCount = sliceCount_new
            sagTSEechoTimeCount = echoTimeCount_new

    
          ###############################
          # T2 MESE scans
          # We want the number of slices to be divisible by 7 because there should be 7 TEs. If not, something is wrong.
          elif ("T2" in ds.SeriesDescription) and ("RIGHT" in ds.SeriesDescription) and (nFiles % 7 == 0):
            if (T2_found == 0):
              T2cubeShape = (int(nFiles/7), (ds.pixel_array).shape[0], (ds.pixel_array).shape[1])    # (z,x,y)
              T2_dx =  float(ds.PixelSpacing[0])
              T2_dy =  float(ds.PixelSpacing[1])
              T2voxpos = np.zeros(T2cubeShape + (3,))
              T2scans = np.zeros((7,) + T2cubeShape)
              # Save the orientation vectors for later interpolating voxel values onto T2 grid.
              T2orientationPatient = ds.ImageOrientationPatient
              T2sliceLocs = np.zeros(T2cubeShape[0])
              T2_found = 1
 
            (sliceCount_new, echoTimeCount_new) = scanInfo(T2scans, ds, T2sliceLocs, sliceCount, T2echoTimes, T2echoTimeCount, T2voxpos, T2cubeShape, T2_dx, T2_dy, T2MinPositionPatient, T2MaxPositionPatient, minmaxT2sliceLocation)
            sliceCount = sliceCount_new
            T2echoTimeCount = echoTimeCount_new
            
    
    if T2_found:
      print("T2 MESE series found")
      print(T2cubeShape)
      print(int(np.floor(T2cubeShape[0]/2)+1))
      # Type in path to T2 maps - see comments to "prepareOAIimageData"
      T2mapPath =
      T2mapPathObj = Path(T2mapPath)
      if T2mapPathObj.is_file():
        print("Saved T2 map found")
        T2map = np.load(T2mapPath)
      else:
        print("Saved T2 map not found")
        # Start computing T2 map
        print('Start computing T2 map')
        X = np.vstack((T2echoTimes,np.ones(7)))
        print("T2cubeShape:")
        print(T2cubeShape)
        T2map = np.zeros((len(T2sliceLocs),T2cubeShape[1],T2cubeShape[2]))
        print(len(T2sliceLocs))
        for zz in range(len(T2sliceLocs)):
          for xx in range(T2cubeShape[1]):
            for yy in range(T2cubeShape[2]):
              y0 = np.log(T2scans[0,zz,xx,yy])
              y1 = np.log(T2scans[1,zz,xx,yy])
              y2 = np.log(T2scans[2,zz,xx,yy])
              y3 = np.log(T2scans[3,zz,xx,yy])
              y4 = np.log(T2scans[4,zz,xx,yy])
              y5 = np.log(T2scans[5,zz,xx,yy])
              y6 = np.log(T2scans[6,zz,xx,yy])
              Y = np.array([y0,y1,y2,y3,y4,y5,y6])
              ab = np.linalg.lstsq(X.T,Y.T)
              T2map[zz,xx,yy] = -1/ab[0][0]        # This will give T2 in ms
        
        np.save(T2mapPath, T2map)

      # Make sure T2 slices are sorted in increasing order.
      T2sortIdx = np.argsort(T2sliceLocs)
      T2sliceLocs = T2sliceLocs[T2sortIdx]
      print(T2sliceLocs)
      print(len(T2sliceLocs))
      T2map = T2map[T2sortIdx,:,:]
      # Find slice spacing
      T2_dz = float(T2sliceLocs[1] - T2sliceLocs[0])
      T2spacings = (T2_dz, T2_dx, T2_dy)

    
    

 
    if SAG_DESS_found and SAG_TSE_found and COR_FLASH_found and COR_TSE_found and T2_found:
      print("DESS, SAG TSE, COR FLASH, COR TSE, and T2 found")
    
      SAG_DESS_scans = np.squeeze(SAG_DESS_scans)
      SAG_TSE_scans = np.squeeze(SAG_TSE_scans)
      COR_TSE_scans = np.squeeze(COR_TSE_scans)
      COR_FLASH_scans = np.squeeze(COR_FLASH_scans)

   
      
      # Make sure DESS slices are sorted in increasing order.
      dessSortIdx = np.argsort(SAG_DESS_sliceLocs)
      SAG_DESS_sliceLocs = SAG_DESS_sliceLocs[dessSortIdx]
      SAG_DESS_scans = SAG_DESS_scans[dessSortIdx,:,:]
      # Need to find the slice spacing. There doesn't seem to be a tag for this 
      # in the dicom so I'll take the difference between two locations.
      dess_dz = float(SAG_DESS_sliceLocs[1] - SAG_DESS_sliceLocs[0])
      sagDESSspacings = (dess_dz, dess_dx, dess_dy)
      
      # Make sure TSE slices are sorted in increasing order.
      sagtseSortIdx = np.argsort(SAG_TSE_sliceLocs)
      SAG_TSE_sliceLocs = SAG_TSE_sliceLocs[sagtseSortIdx]
      SAG_TSE_scans = SAG_TSE_scans[sagtseSortIdx,:,:]
      # Need to find the slice spacing. There doesn't seem to be a tag for this 
      # in the dicom so I'll take the difference between two locations.
      sagtse_dz = float(SAG_TSE_sliceLocs[1] - SAG_TSE_sliceLocs[0])
      sagTSEspacings = (sagtse_dz, sagtse_dx, sagtse_dy)
      
      # Make sure COR FLASH slices are sorted in increasing order.
      corflashSortIdx = np.argsort(COR_FLASH_sliceLocs)
      COR_FLASH_sliceLocs = COR_FLASH_sliceLocs[corflashSortIdx]
      COR_FLASH_scans = COR_FLASH_scans[corflashSortIdx,:,:]
      
      # Make sure COR TSE slices are sorted in increasing order.
      cortseSortIdx = np.argsort(COR_TSE_sliceLocs)
      COR_TSE_sliceLocs = COR_TSE_sliceLocs[cortseSortIdx]
      COR_TSE_scans = COR_TSE_scans[cortseSortIdx,:,:]

      # Produce an array with all the x,y,z coordinates of the T2 acquisitions along the column dimension.
      P_T2_3D = np.array([ (T2voxpos[:,:,:,0]).flatten(), 
                           (T2voxpos[:,:,:,1]).flatten(), 
                           (T2voxpos[:,:,:,2]).flatten(), 
                           np.ones(T2cubeShape[0]*T2cubeShape[1]*T2cubeShape[2])])


      dessValuesInT2 = interPolateToT2grid(SAG_DESS_scans, sagDESScubeShape, dessMinPositionPatient, dessMaxPositionPatient, 
                                           dessOrientationPatient, dess_dx, dess_dy, P_T2_3D)
      dessValuesInT2_reshaped = dessValuesInT2.reshape((T2cubeShape[0],T2cubeShape[1],T2cubeShape[2]))
      
      # Normalization - make DESS *volume* have zero mean and unit variance. 
      dessValuesInT2_reshaped = (dessValuesInT2_reshaped - np.mean(dessValuesInT2_reshaped))/np.std(dessValuesInT2_reshaped)



      # Print out slices. We now have values for DESS, sag/cor TSE, and FLASH that are interpolated to the T2 grid and can write them out directly.
      #for ll in range(len(T2sliceLocs)):
      for ll in range(1, len(T2sliceLocs)-1):
        

      #  # Current slice
         DESS_slice1 = dessValuesInT2_reshaped[ll,:,:]
         if not os.path.exists(outputDir + patientPath):
           os.makedirs(outputDir + patientPath)
         generatedImages += 1
         imName = outputDir + patientPath + "/" + str(generatedImages).zfill(4) + ".h5"
         hf = h5py.File(imName, 'w')
         hf.create_dataset('data', data=DESS_slice1)
         hf.close()

  # Done with this patient
  print("***************")

#
