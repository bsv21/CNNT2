# This script combines the anatomical images, created with "prepareOAIimageData", with the cartilage masks,
# created with "prepareDESSforMasks" and "createMasksForCNNT2". The combined data is then processed with the
# CNNT2proc program.

import os
from PIL import Image

# Path to your anatomical 1x5 image files
baseImPath = 
# Path to your cartilage mask files
baseMaskPath = 
# Path to write your output
outputPath = 
fileList = []
patientDirList = []
for (patientDirpath, patientDirnames, patientFilenames) in os.walk(baseImPath):
  patientDirList.extend(patientDirnames)
  break
patientDirList.sort()
nPatients = int(len(patientDirList))

for patientPath in patientDirList:
  fList = []
  for (dirpath, dirnames, filenames) in os.walk(baseImPath + '/' + patientPath):
    fList.extend(filenames)
    break
  fList.sort()
  patientFileList = [patientPath + '/' + file_n for file_n in fList]
  fileList.extend(patientFileList)
nFiles = int(len(fileList))

# Filelist should now contain a list of strings of type [patientNo]/[sliceNo].jpg,
# e.g., 9999865/0017.jpg.
# The "ims" directory (with OAI images) and "masks" directory (with deep learning drawn masks) should both contain these jpgs.
generatedIms = 0
for jpgPath in fileList:
  generatedIms += 1
  print(generatedIms)
  oaiIm = Image.open(baseImPath + "/" + jpgPath) 
  oaiWidth, oaiHeight = oaiIm.size
  mask = Image.open(baseMaskPath + "/" + jpgPath)
  maskCropped = mask.crop((48,48,336,288))
  jpgPathSplit = jpgPath.split("/")
  patientStr = jpgPathSplit[0]
  new_im = Image.new('L', (int(oaiWidth/5*6), oaiHeight))
  new_im.paste(oaiIm)
  new_im.paste(maskCropped,(oaiWidth,0))
  new_im.paste(maskCropped,(oaiWidth,int(oaiHeight/3)))
  new_im.paste(maskCropped,(oaiWidth,int(oaiHeight/3*2)))
  new_im.save(outputPath + "/" + str(generatedIms).zfill(6) + '_' + patientStr + '.jpg')
