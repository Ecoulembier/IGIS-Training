# MsProject2
This repository contains image pre-processing, augmentation and neural network training and validation scripts.

## Pipeline process

1) **PreProcessing**  
This folder contains image preparation scripts.
- *PreProcessImages.py*: Crop all images to precisely contain the petri dish for further processing.
- *Plate2Plant_Macro.java*: A macro to be used in ImageJ FIJI. This script takes 12 drawn boxes coordinates and returns a text file with the location of each box.
- *Plate2PlantsProcessing.py*: This script takes the coordinates of the boxes in the created text file and crops all individual plants out of the images according
  to the corresponding plate number, since these positions are fixed.
After the preprocessing, all images were annotated using the online tool *Supervisely*. Link: https://app.supervisely.com

2) **Augmentations**  
Here, all augmentations performed on the annotated dataset are listed (4), with 2 additional scripts.
- *Augment_\*.py*: The 4 augmentation script used to enrich the training data set size.
- *image_padding.py*: This script performs symmetric padding to each image for further consistency across the training process and to retain the relative plant
  surface areas.
- *check_masks.py*: This script was used to visualize the machine mask images as they just show as black images for humans. This makes the mask visual as a red
  object of interest against a black background.

3) **train-validation**  
This folder contains the neural network training and validation script.
- *trainCompleteV2.py*: This script trains a DeepLapV3+ model with a pre-trained ResNet50 backbone on the created training set, outputting model checkpoints every
  5 epochs. Additionally, every epoch, metrics such as the loss, IoU, precision and recall are printed out to console (or an output file).
- *validateComplete.py*: This script takes the location of a previously trained model checkpoint and validates the model using the validation set, outputting a
  csv file which holds the average (and individual) IoU, precision and recall values. Reflecting real use, visual predictions on the image segmentations are also
  outputted in a separate folder, with another file containing the plant surfaces in a text file.
