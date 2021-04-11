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

# Specify a directory containing OAI data /path/to/directory/. This directory should contain many subdirectories,
# each representing an OAI subject.
basePath =
# Specify a directory where the prepared output images get saved. They are not saved in separate directories.
outputDir =
# patientDirList will contain the names of all the different directories in 'basepath', each representing a subject.
patientDirList = []
# patientCount will contain the number of subjects.
patientCount = 0


# The function scanInfo looks at a single slice in a sequence, compares it to what has been computed for previous slices for the same sequence, and
# updates the number of slices and the number of echo times if appropriate.
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


# The function 'interPolateToT2grid' takes a sequence and resamples it so it matches the locations in the
# T2/MESE images.
def interPolateToT2grid(scanData, cubeShape, minPosPatient, maxPosPatient, orientationPatient, dx, dy, PT2_3D):
  # The vector zVec will should represent the orientation of the sequence z-axis in the scanner coordinates.
  zVec = [minPosPatient[0] - maxPosPatient[0], minPosPatient[1] - maxPosPatient[1], minPosPatient[2] - maxPosPatient[2]]

  # Compute matrix to transform 3D dataset from T2 index frame to scanner physical coordinates frame.
  A = np.array([ [orientationPatient[3]*dx, orientationPatient[0]*dy, zVec[0]/(1-cubeShape[0]), minPosPatient[0]],
                 [orientationPatient[4]*dx, orientationPatient[1]*dy, zVec[1]/(1-cubeShape[0]), minPosPatient[1]],
                 [orientationPatient[5]*dx, orientationPatient[2]*dy, zVec[2]/(1-cubeShape[0]), minPosPatient[2]],
                 [0, 0, 0, 1] ])
 
  # Now find the solution to AxC_T2 = PT2_3D
  C_T2 = np.linalg.solve(A, PT2_3D)
  
  # Create an interpolator object for the sequence and its dimensions.    
  interp = RegularGridInterpolator((np.arange(cubeShape[1]), np.arange(cubeShape[2]), np.arange(cubeShape[0])),np.transpose(scanData,(1,2,0)), bounds_error=False, fill_value=0)
  # Use the interpolator and our calculated C_T2 matrix to obtain the interpolation of the sequence data in the T2 grid (resampling the data to match the T2 pixels).
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
  
  # Loop through the exam dates for this patient.
  for datePath in dateDirList:
    seqCount += 1
    print(datePath + "\n")
    
    # Make a list of all the sequences performed on the subject on that particular date.
    for (sequenceDirpath, sequenceDirnames, sequenceFilenames) in os.walk(basePath + patientPath + "/" + datePath + "/"):
    	sequenceDirList.extend(sequenceDirnames)
    	break
    sequenceDirList.sort()
    print(sequenceDirList)

    # We will loop through all the sequences for this subject. We will check whether they are described
    # as Cor TSE, Sag DESS, Cor Flash, Sag TSE, or MESE. If they are, we will store them in data arrays
    # along with some imaging parameters.
    # We start by initializing those imaging parameters, either with zeros or with dummy numbers.    
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
   
    # Now loop through all the sequences found for this subject on this date. 
    for seq in sequenceDirList:
    
        sliceCount = 0
   
        # For each sequence, we create a list of all the files in its directory.
        # These should be DICOMs, representing slices. 
        fileList = []
        sequencepath = sequenceDirpath + seq + "/"
        print(sequencepath)
        for (dirpath, dirnames, filenames) in os.walk(sequencepath):
            fileList.extend(filenames)
            break
        fileList.sort()
        # The number of files found, representing number of slices for this sequence.
        nFiles = int(len(fileList))
        # The sliceLocs array will contain the different slice locations, we initialize it to zero.
        sliceLocs = np.zeros(nFiles)
    
        # Now loop through all the dicoms for this sequence on this date for this subject 
        for ff in fileList:
          dicomPath = sequencepath + ff
          # ds is the DICOM (contains data and header info)
          ds = pydicom.dcmread(dicomPath);
          
          ###############################
          # COR FLASH scans
          # If the following statement is true, we have found a FLASH scan in the right knee.
          if ("FLASH" in ds.SeriesDescription) and ("RIGHT" in ds.SeriesDescription):
            # If we have not read a Cor FLASH DICOM before, we read its header to get the scan
            # settings that was used in the FLASH scans.
            if (COR_FLASH_found == 0):
              # Thinking of the FLASH scan volume as a 3D cube, we store its dimensions
              corFLASHcubeShape = (nFiles, (ds.pixel_array).shape[0], (ds.pixel_array).shape[1])    # (z,x,y)
              # Store the FLASH voxel width in x
              corflash_dx =  float(ds.PixelSpacing[0])
              # Store the FLASH voxel width in y
              corflash_dy =  float(ds.PixelSpacing[1])
              # The voxpos array will contain the voxel positions of the scan, with an added dimension to perform a coordinate transformation.
              corFLASHvoxpos = np.zeros(corFLASHcubeShape + (3,))
              # Will store the DICOM image data in COR_FLASH_scans. Adding a dimension for echo times, but this is only used for the MESE scans and
              # kept for consistency with the other scans.
              COR_FLASH_scans = np.zeros((1,) + corFLASHcubeShape)
              # Save the orientation vectors for later interpolating voxel values onto T2 grid.
              corflashOrientationPatient = ds.ImageOrientationPatient
              # The sliceLocs array will contain the different slice locations, we initialize it to zero.
              COR_FLASH_sliceLocs = np.zeros(nFiles)
              # Now note that we have read at least one Cor FLASH DICOM, so we don't need to read its header info again.
              COR_FLASH_found = 1

            # The 'scanInfo' function adds the current slice to the COR_FLASH_scans structure, that will eventually contain all the image data for the sequence.
            # Also, if the DICOM that is currently being read corresponds to a different echo time than previous DICOMS from the same sequence,
            # we add it to the list of echo times for this sequence.
            # This was done to be flexible in case there were any discrepancies in the number of echo times for the different sequences,
            # but this is actually very constant and the number of echo times could be assumed to be 1 for all sequences except MESE, which has 7. 
            (sliceCount_new, echoTimeCount_new) = scanInfo(COR_FLASH_scans, ds, COR_FLASH_sliceLocs, sliceCount, corFLASHechoTimes, corFLASHechoTimeCount, corFLASHvoxpos, corFLASHcubeShape, corflash_dx, corflash_dy, corflashMinPositionPatient, corflashMaxPositionPatient, minmaxCorFLASHsliceLocation)
            # I don't think this sliceCount variable is used anymore - should delete
            sliceCount = sliceCount_new
            # Add to number of echo times if necessary - this should never exceed 1 except for MESE.
            corFLASHechoTimeCount = echoTimeCount_new
          
          
          ###############################
          # SAG DESS scans
          # If the following statement is true, we have found a DESS scan in the right knee.
          if ("DESS" in ds.SeriesDescription) and ("RIGHT" in ds.SeriesDescription):
            # If we have not read a Sag DESS DICOM before, we read its header to get the scan
            # settings that was used in the Sag DESS scans.
            if (SAG_DESS_found == 0):
              # Thinking of the DESS scan volume as a 3D cube, we store its dimensions
              sagDESScubeShape = (nFiles, (ds.pixel_array).shape[0], (ds.pixel_array).shape[1])    # (z,x,y)
              # Store the DESS voxel width in x
              dess_dx =  float(ds.PixelSpacing[0])
              # Store the DESS voxel width in y
              dess_dy =  float(ds.PixelSpacing[1])
              # The voxpos array will contain the voxel positions of the scan, with an added dimension to perform a coordinate transformation.
              sagDESSvoxpos = np.zeros(sagDESScubeShape + (3,))
              # Will store the DICOM image data in COR_FLASH_scans. Adding a dimension for echo times, but this is only used for the MESE scans and
              # kept for consistency with the other scans.
              SAG_DESS_scans = np.zeros((1,) + sagDESScubeShape)
              # Save the orientation vectors for later interpolating voxel values onto T2 grid.
              dessOrientationPatient = ds.ImageOrientationPatient
              # The sliceLocs array will contain the different slice locations, we initialize it to zero.
              SAG_DESS_sliceLocs = np.zeros(nFiles)
              # Now note that we have read at least one Sag DESS DICOM, so we don't need to read its header info again.
              SAG_DESS_found = 1
 
            # The 'scanInfo' function adds the current slice to the SAG_DESS_scans structure, that will eventually contain all the image data for the sequence.
            # Also, if the DICOM that is currently being read corresponds to a different echo time than previous DICOMS from the same sequence,
            # we add it to the list of echo times for this sequence.
            # This was done to be flexible in case there were any discrepancies in the number of echo times for the different sequences,
            # but this is actually very constant and the number of echo times could be assumed to be 1 for all sequences except MESE, which has 7. 
            (sliceCount_new, echoTimeCount_new) = scanInfo(SAG_DESS_scans, ds, SAG_DESS_sliceLocs, sliceCount, sagDESSechoTimes, sagDESSechoTimeCount, sagDESSvoxpos, sagDESScubeShape, dess_dx, dess_dy, dessMinPositionPatient, dessMaxPositionPatient, minmaxDESSsliceLocation)
            # I don't think this sliceCount variable is used anymore - should delete
            sliceCount = sliceCount_new
            # Add to number of echo times if necessary - this should never exceed 1 except for MESE.
            sagDESSechoTimeCount = echoTimeCount_new
    
    
          ###############################
          # COR TSE scans
          # If the following statement is true, we have found a Cor TSE scan in the right knee.
          if ("COR" in ds.SeriesDescription) and ("TSE" in ds.SeriesDescription) and ("RIGHT" in ds.SeriesDescription):
            # If we have not read a Cor TSE DICOM before, we read its header to get the scan
            # settings that was used in the Cor TSE scans.
            if (COR_TSE_found == 0):
              # Thinking of the Cor TSE scan volume as a 3D cube, we store its dimensions
              corTSEcubeShape = (nFiles, (ds.pixel_array).shape[0], (ds.pixel_array).shape[1])    # (z,x,y)
              # Store the Cor TSE voxel width in x
              cortse_dx =  float(ds.PixelSpacing[0])
              # Store the Cor TSE voxel width in y
              cortse_dy =  float(ds.PixelSpacing[1])
              # The voxpos array will contain the voxel positions of the scan, with an added dimension to perform a coordinate transformation.
              corTSEvoxpos = np.zeros(corTSEcubeShape + (3,))
              # Will store the DICOM image data in COR_FLASH_scans. Adding a dimension for echo times, but this is only used for the MESE scans and
              # kept for consistency with the other scans.
              COR_TSE_scans = np.zeros((1,) + corTSEcubeShape)
              # Save the orientation vectors for later interpolating voxel values onto T2 grid.
              cortseOrientationPatient = ds.ImageOrientationPatient
              # The sliceLocs array will contain the different slice locations, we initialize it to zero.
              COR_TSE_sliceLocs = np.zeros(nFiles)
              # Now note that we have read at least one Cor TSE DICOM, so we don't need to read its header info again.
              COR_TSE_found = 1

            # The 'scanInfo' function adds the current slice to the COR_TSE_scans structure, that will eventually contain all the image data for the sequence.
            # Also, if the DICOM that is currently being read corresponds to a different echo time than previous DICOMS from the same sequence,
            # we add it to the list of echo times for this sequence.
            # This was done to be flexible in case there were any discrepancies in the number of echo times for the different sequences,
            # but this is actually very constant and the number of echo times could be assumed to be 1 for all sequences except MESE, which has 7. 
            (sliceCount_new, echoTimeCount_new) = scanInfo(COR_TSE_scans, ds, COR_TSE_sliceLocs, sliceCount, corTSEechoTimes, corTSEechoTimeCount, corTSEvoxpos, corTSEcubeShape, cortse_dx, cortse_dy, cortseMinPositionPatient, cortseMaxPositionPatient, minmaxCorTSEsliceLocation)
            # I don't think this sliceCount variable is used anymore - should delete
            sliceCount = sliceCount_new
            # Add to number of echo times if necessary - this should never exceed 1 except for MESE.
            corTSEechoTimeCount = echoTimeCount_new

    
          ###############################
          # SAG TSE scans
          # If the following statement is true, we have found a Sag TSE scan in the right knee.
          if ("SAG" in ds.SeriesDescription) and ("TSE" in ds.SeriesDescription) and ("RIGHT" in ds.SeriesDescription):
            # If we have not read a Sag TSE DICOM before, we read its header to get the scan
            # settings that was used in the Sag TSE scans.
            if (SAG_TSE_found == 0):
              # Thinking of the Sag TSE scan volume as a 3D cube, we store its dimensions
              sagTSEcubeShape = (nFiles, (ds.pixel_array).shape[0], (ds.pixel_array).shape[1])    # (z,x,y)
              # Store the Sag TSE voxel width in x
              sagtse_dx =  float(ds.PixelSpacing[0])
              # Store the Sag TSE voxel width in y
              sagtse_dy =  float(ds.PixelSpacing[1])
              # The voxpos array will contain the voxel positions of the scan, with an added dimension to perform a coordinate transformation.
              sagTSEvoxpos = np.zeros(sagTSEcubeShape + (3,))
              # Will store the DICOM image data in SAG_FLASH_scans. Adding a dimension for echo times, but this is only used for the MESE scans and
              # kept for consistency with the other scans.
              SAG_TSE_scans = np.zeros((1,) + sagTSEcubeShape)
              # Save the orientation vectors for later interpolating voxel values onto T2 grid.
              sagtseOrientationPatient = ds.ImageOrientationPatient
              # The sliceLocs array will contain the different slice locations, we initialize it to zero.
              SAG_TSE_sliceLocs = np.zeros(nFiles)
              # Now note that we have read at least one Sag TSE DICOM, so we don't need to read its header info again.
              SAG_TSE_found = 1
 
            # The 'scanInfo' function adds the current slice to the SAG_TSE_scans structure, that will eventually contain all the image data for the sequence.
            # Also, if the DICOM that is currently being read corresponds to a different echo time than previous DICOMS from the same sequence,
            # we add it to the list of echo times for this sequence.
            # This was done to be flexible in case there were any discrepancies in the number of echo times for the different sequences,
            # but this is actually very constant and the number of echo times could be assumed to be 1 for all sequences except MESE, which has 7. 
            (sliceCount_new, echoTimeCount_new) = scanInfo(SAG_TSE_scans, ds, SAG_TSE_sliceLocs, sliceCount, sagTSEechoTimes, sagTSEechoTimeCount, sagTSEvoxpos, sagTSEcubeShape, sagtse_dx, sagtse_dy, sagtseMinPositionPatient, sagtseMaxPositionPatient, minmaxSagTSEsliceLocation)
            # I don't think this sliceCount variable is used anymore - should delete
            sliceCount = sliceCount_new
            # Add to number of echo times if necessary - this should never exceed 1 except for MESE.
            sagTSEechoTimeCount = echoTimeCount_new

    
          ###############################
          # T2 MESE scans
          # If the following statement is true, we have found a T2 MESE scan in the right knee.
          # We want the number of slices to be divisible by 7 because there should be 7 TEs. If not, something is wrong.
          elif ("T2" in ds.SeriesDescription) and ("RIGHT" in ds.SeriesDescription) and (nFiles % 7 == 0):
            # If we have not read a T2 MESE DICOM before, we read its header to get the scan
            # settings that was used in the T2 MESE scans.
            if (T2_found == 0):
              # Thinking of the T2 MESE scan volume as a 3D cube, we store its dimensions.
              # We divide by 7 since we have 7 TEs.
              T2cubeShape = (int(nFiles/7), (ds.pixel_array).shape[0], (ds.pixel_array).shape[1])    # (z,x,y)
              # Store the T2 MESE voxel width in x
              T2_dx =  float(ds.PixelSpacing[0])
              # Store the T2 MESE voxel width in y
              T2_dy =  float(ds.PixelSpacing[1])
              # The voxpos array will contain the voxel positions of the scan, with an added dimension to perform a coordinate transformation.
              T2voxpos = np.zeros(T2cubeShape + (3,))
              # Will store the DICOM image data in SAG_FLASH_scans. Adding a dimension for the 7 echo times.
              T2scans = np.zeros((7,) + T2cubeShape)
              # Save the orientation vectors for later interpolating voxel values onto T2 grid.
              T2orientationPatient = ds.ImageOrientationPatient
              # The sliceLocs array will contain the different slice locations, we initialize it to zero.
              T2sliceLocs = np.zeros(T2cubeShape[0])
              # Now note that we have read at least one T2 MESE DICOM, so we don't need to read its header info again.
              T2_found = 1
 
            # The 'scanInfo' function adds the current slice to the SAG_TSE_scans structure, that will eventually contain all the image data for the sequence.
            # Also, if the DICOM that is currently being read corresponds to a different echo time than previous DICOMS from the same sequence,
            # we add it to the list of echo times for this sequence.
            # This was done to be flexible in case there were any discrepancies in the number of echo times for the different sequences,
            # but this is actually very constant and the number of echo times could be assumed to be 1 for all sequences except MESE, which has 7. 
            (sliceCount_new, echoTimeCount_new) = scanInfo(T2scans, ds, T2sliceLocs, sliceCount, T2echoTimes, T2echoTimeCount, T2voxpos, T2cubeShape, T2_dx, T2_dy, T2MinPositionPatient, T2MaxPositionPatient, minmaxT2sliceLocation)
            # I don't think this sliceCount variable is used anymore - should delete
            sliceCount = sliceCount_new
            # Add to number of echo times if necessary - the MESE echo times should have a total of 7.
            T2echoTimeCount = echoTimeCount_new
            
   
    # We now proceed to form a T2 map by doing an exponential fit of the 7 MESE acquisitions, each with its own echo time. 
    if T2_found:
      print("T2 MESE series found")
      # Now specify a path to a numpy file where the T2 fit is stored. Should by of the form
      # /path/to/files/t2File_" + patientPath + ".npy"
      T2mapPath = 
      T2mapPathObj = Path(T2mapPath)
      # Sometimes, we will already have computed the T2 fit for this scan (for example, when you start the script,
      # then stop it and then continue it later but let it loop through all the files again. In those cases, we don't
      # recompute the T2 map, which saves time.
      if T2mapPathObj.is_file():
        print("Saved T2 map found")
        T2map = np.load(T2mapPath)
      else:
        # We did not find the T2 map for this scan, so we compute it
        print("Saved T2 map not found")
        # Start computing T2 map
        print('Start computing T2 map')
        X = np.vstack((T2echoTimes,np.ones(7)))
        T2map = np.zeros((len(T2sliceLocs),T2cubeShape[1],T2cubeShape[2]))
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
      T2map = T2map[T2sortIdx,:,:]
      # Find slice spacing
      T2_dz = float(T2sliceLocs[1] - T2sliceLocs[0])
      T2spacings = (T2_dz, T2_dx, T2_dy)

    
    

    # Now proceed with the interpolation. We only proceed if we have found all the scans. 
    if SAG_DESS_found and SAG_TSE_found and COR_FLASH_found and COR_TSE_found and T2_found:
      print("DESS, SAG TSE, COR FLASH, COR TSE, and T2 found")
   
      # Remove the extra dimension from the non-MESE scans - should fix this. 
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
      #corflashSortIdx = np.flip(corflashSortIdx,0)
      COR_TSE_sliceLocs = COR_TSE_sliceLocs[cortseSortIdx]
      COR_TSE_scans = COR_TSE_scans[cortseSortIdx,:,:]

      # Produce an array with all the x,y,z coordinates of the T2 acquisitions along the column dimension.
      P_T2_3D = np.array([ (T2voxpos[:,:,:,0]).flatten(), 
                           (T2voxpos[:,:,:,1]).flatten(), 
                           (T2voxpos[:,:,:,2]).flatten(), 
                           np.ones(T2cubeShape[0]*T2cubeShape[1]*T2cubeShape[2])])


      # Now interpolate the DESS values to the T2 grid  using the 'interPolateToT2grid' function defined above.
      dessValuesInT2 = interPolateToT2grid(SAG_DESS_scans, sagDESScubeShape, dessMinPositionPatient, dessMaxPositionPatient, 
                                           dessOrientationPatient, dess_dx, dess_dy, P_T2_3D)
      # Reshape the result to have the dimensions of the T2 scan.
      dessValuesInT2_reshaped = dessValuesInT2.reshape((T2cubeShape[0],T2cubeShape[1],T2cubeShape[2]))
      

      # Interpolate the Sag TSE values to the T2 grid  using the 'interPolateToT2grid' function defined above.
      sagtseValuesInT2 = interPolateToT2grid(SAG_TSE_scans, sagTSEcubeShape, sagtseMinPositionPatient, sagtseMaxPositionPatient, 
                                             sagtseOrientationPatient, sagtse_dx, sagtse_dy, P_T2_3D)
      # Reshape the result to have the dimensions of the T2 scan.
      sagtseValuesInT2_reshaped = sagtseValuesInT2.reshape((T2cubeShape[0],T2cubeShape[1],T2cubeShape[2]))
      

      # Interpolate the FLASH values to the T2 grid  using the 'interPolateToT2grid' function defined above.
      corflashValuesInT2 = interPolateToT2grid(COR_FLASH_scans, corFLASHcubeShape, corflashMinPositionPatient, corflashMaxPositionPatient, 
                                               corflashOrientationPatient, corflash_dx, corflash_dy, P_T2_3D)
      # Reshape the result to have the dimensions of the T2 scan.
      corflashValuesInT2_reshaped = corflashValuesInT2.reshape((T2cubeShape[0],T2cubeShape[1],T2cubeShape[2]))


      # Interpolate the Cor TSE values to the T2 grid  using the 'interPolateToT2grid' function defined above.
      cortseValuesInT2 = interPolateToT2grid(COR_TSE_scans, corTSEcubeShape, cortseMinPositionPatient, cortseMaxPositionPatient, 
                                               cortseOrientationPatient, cortse_dx, cortse_dy, P_T2_3D)
      # Reshape the result to have the dimensions of the T2 scan.
      cortseValuesInT2_reshaped = cortseValuesInT2.reshape((T2cubeShape[0],T2cubeShape[1],T2cubeShape[2]))




      # Print out slices. We now have values for DESS, sag/cor TSE, and FLASH that are interpolated to the T2 grid and can write them out directly.
      #for ll in range(len(T2sliceLocs)):
      for ll in range(1, len(T2sliceLocs)-1):
        

        # We "zoom in" to a 240x288 area that contains the knee joint. This was done both to reduce the size of the image data and also to
        # decrease waste in the network fitting, as having a lot of empty regions would just be asking the network to fit noise to noise.
        # These precise values were found by trial-and-error, but seemed to work on dozens of data sets.
        y_m = 48
        y_M = 288
        x_m = 48
        x_M = 336
        # Perform masking. We throw away anything that is less than 0.15x or bigger than 2x 
        # the maximum value in the center of the center slice of the MESE scan with the lowest TE.
        # (The 2x condition gets rid of some aliasing artifacts in the OAI scans).
        zeroMaskT2 = np.ones(T2map.shape)
        zeroMaskT2[T2scans[0,:,:,:] < 0.15*np.amax(T2scans[0,int(np.floor(T2cubeShape[0]/2)+1),120:270,120:270])] = 0
        zeroMaskT2[T2scans[0,:,:,:] > 2*np.amax(T2scans[0,int(np.floor(T2cubeShape[0]/2)+1),120:270,120:270])] = 0
        T2map_masked = T2map*zeroMaskT2
        T2_slice_masked0 = T2map_masked[ll-1,y_m:y_M,x_m:x_M]
        T2_slice_masked1 = T2map_masked[ll,y_m:y_M,x_m:x_M]
        T2_slice_masked2 = T2map_masked[ll+1,y_m:y_M,x_m:x_M]
       
        # We now have 5 images, and for each we collect slice N-1,N,N+1, resulting in 3x5 images. Later, the cartilage
        # mask should be added to this. 
        # Slice before current slice
        DESS_slice0 = dessValuesInT2_reshaped[ll-1,y_m:y_M,x_m:x_M]
        sagTSE_slice0 = sagtseValuesInT2_reshaped[ll-1,y_m:y_M,x_m:x_M]
        FLASH_slice0 = corflashValuesInT2_reshaped[ll-1,y_m:y_M,x_m:x_M]
        corTSE_slice0 = cortseValuesInT2_reshaped[ll-1,y_m:y_M,x_m:x_M]
        # Scale to [0,255]
        DESS_slice0 = DESS_slice0 * 255.0/np.amax(DESS_slice0)
        sagTSE_slice0 = sagTSE_slice0 * 255.0/np.amax(sagTSE_slice0)
        FLASH_slice0 = FLASH_slice0 * 255.0/np.amax(FLASH_slice0)
        corTSE_slice0 = corTSE_slice0 * 255.0/np.amax(corTSE_slice0)
        # Current slice
        DESS_slice1 = dessValuesInT2_reshaped[ll,y_m:y_M,x_m:x_M]
        sagTSE_slice1 = sagtseValuesInT2_reshaped[ll,y_m:y_M,x_m:x_M]
        FLASH_slice1 = corflashValuesInT2_reshaped[ll,y_m:y_M,x_m:x_M]
        corTSE_slice1 = cortseValuesInT2_reshaped[ll,y_m:y_M,x_m:x_M]
        # Scale to [0,255]
        DESS_slice1 = DESS_slice1 * 255.0/np.amax(DESS_slice1)
        sagTSE_slice1 = sagTSE_slice1 * 255.0/np.amax(sagTSE_slice1)
        FLASH_slice1 = FLASH_slice1 * 255.0/np.amax(FLASH_slice1)
        corTSE_slice1 = corTSE_slice1 * 255.0/np.amax(corTSE_slice1)
        # Slice after current slice
        DESS_slice2 = dessValuesInT2_reshaped[ll+1,y_m:y_M,x_m:x_M]
        sagTSE_slice2 = sagtseValuesInT2_reshaped[ll+1,y_m:y_M,x_m:x_M]
        FLASH_slice2 = corflashValuesInT2_reshaped[ll+1,y_m:y_M,x_m:x_M]
        corTSE_slice2 = cortseValuesInT2_reshaped[ll+1,y_m:y_M,x_m:x_M]
        # Scale to [0,255]
        DESS_slice2 = DESS_slice2 * 255.0/np.amax(DESS_slice2)
        sagTSE_slice2 = sagTSE_slice2 * 255.0/np.amax(sagTSE_slice2)
        FLASH_slice2 = FLASH_slice2 * 255.0/np.amax(FLASH_slice2)
        corTSE_slice2 = corTSE_slice2 * 255.0/np.amax(corTSE_slice2)


        # Concatenate the 5 different images for slice N-1. Later the cartilage mask should be added at the end.
        combinedImages0 = np.concatenate((T2_slice_masked0, DESS_slice0, sagTSE_slice0, FLASH_slice0, corTSE_slice0), axis=1)
        # Concatenate the 5 different images for slice N. Later the cartilage mask should be added at the end.
        combinedImages1 = np.concatenate((T2_slice_masked1, DESS_slice1, sagTSE_slice1, FLASH_slice1, corTSE_slice1), axis=1)
        # Concatenate the 5 different images for slice N+1. Later the cartilage mask should be added at the end.
        combinedImages2 = np.concatenate((T2_slice_masked2, DESS_slice2, sagTSE_slice2, FLASH_slice2, corTSE_slice2), axis=1)
        # We now have 3 rows of 5 images, representing the 3 slices. Now concatenate them together into a 3x5 image matrix.
        combinedImages = np.concatenate((combinedImages0, combinedImages1, combinedImages2), axis=0)
        # Same image matrix to the output directory as a jpg.
        im = Image.fromarray(combinedImages)
        im = im.convert("L")
        generatedImages += 1
        imName = outputDir + patientPath + "/" + str(generatedImages).zfill(4) + ".jpg"
        if not os.path.exists(outputDir + patientPath):
          os.makedirs(outputDir + patientPath)
        im.save(imName)

  # Done with this patient
  print("***************")

