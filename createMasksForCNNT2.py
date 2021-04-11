# This script creates femoral cartilage masks using the network presented by http://arxiv.org/abs/1902.01977.
# Please see the paper above for the most recent version of that code.

import os
import numpy as np
import h5py
from utils.generator_msk_seg import img_generator_oai_bsv
from utils.models import unet_2d_model
from PIL import Image
from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "3"



test_batch_size = 10
img_size = (384,384,1)
model_weights = '../unet_2d_fc_weights--0.8968.h5'
K.set_image_data_format('channels_last')
# create the unet model
model = unet_2d_model(img_size)
model.load_weights(model_weights);

# Decide where to put the masks
outputBaseDir =

bb = 0
for imageData, imageNames in img_generator_oai_bsv(test_batch_size, img_size):
  print('bb: ' + str(bb))
  bb += 1
  recon = model.predict(imageData, batch_size = test_batch_size)  #shape: (batch_size,xdim,ydim,1)
  for ii in range(recon.shape[0]):
    print("max recon:")
    print(np.amax(recon[ii,:,:,0]))
    recscaled = recon[ii,:,:,0]
    recscaled = recscaled*255.0/np.amax(recscaled)
    im = Image.fromarray(recscaled)
    print("max im:")
    print(np.amax(im))
    im = im.convert("L")
    imNameSplit_ii = imageNames[ii].split(".")
    imName_ii = imNameSplit_ii[0]
    dirNameSplit_ii = imageNames[ii].split("/")
    dirName_ii = dirNameSplit_ii[0]
    imNameFull = outputBaseDir + "/" + imName_ii + ".jpg"
    if not os.path.exists(outputBaseDir + "/" + dirName_ii):
      os.makedirs(outputBaseDir + "/" + dirName_ii)
    im.save(imNameFull) 
