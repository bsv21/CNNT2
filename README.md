# CNNT2

Code used for Radiology: Artificial Intelligence paper "Synthesizing Quantitative T2 Maps in Right Lateral Knee Femoral Condyles from Multi-Contrast Anatomical Data with a Conditional GAN" by Sveinsson et al.

1. The "prepareOAIimageData" script takes in OAI DICOMs and returns jpegs, each one representing an MESE T2 slice and other OAI sequences resampled to align with the MESE, combined into one jpeg.

2. The "prepareDESSforMasks" program takes in OAI DESS DICOMs and converts them into h5 files. Very similar to the prepareOAIimageData script. The "createMasksForCNNT2" script then uses these h5 files to create catilage masks in the form of jpegs. It does this using a network from a separate project.

3. The jpegs from 1 and 2 are combined using the script "combineImsWithMasks", resulting in jpegs that contain both the slices from all the different sequences as well as the cartilage mask.

4. The jpegs from 3 are then used in a neural network using the script "CNNT2proc".
