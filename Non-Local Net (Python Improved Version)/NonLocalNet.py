## Create and Train Non-Local Neural Network Model for Recognition
# Former Author: nanting03, JoeyBG;
# Improved By: JoeyBG;
# Affiliation: Beijing Institute of Technology, Radar Research Lab;
# Date: 2023-9-1;
# Language & Platform: Python, Paddlepaddle Framework.
#
# Introduction:
# The following code is a Python implementation of an improved version of 
# Non-Local Net using PaddlePaddle. It is based on the MATLAB version of the 
# code and incorporates certain upgrades, potentially leading to improved 
# accuracy, robustness, and generalization capabilities.
#
# The codes in NonLocalNet_Codes are in different functions showing in the table below:
# --------------------------------------------------------------------------------
#   Names of the codes	              Functions
# --------------------------------------------------------------------------------
#   non_local.py	                  Definition of Non-Local Modules Part. 1
#   context_block.py	              Definition of Non-Local Modules Part. 2
#   nlnet.py	                      Definition of the Network
#   radar_har_dataset.py	          Definition of the Dataset
#   config.py	                      Configurations
#   train_val_split.py	              Splitting the Training and Validation Sets
#   train.py                          Start Model Training
#   eval.py	                          Start Model Validation
# --------------------------------------------------------------------------------
#
# P.S. The datas in your work/dataset should be in the structure of:
# The name of dir is the labels.
# The images in dir should contain both training and validation datas (In another word, all datas).
# The program will randomly cut the datas into training set and validation set automatically.

# Import necessary libraries for image loading.
import os
import random
from matplotlib import pyplot as plt
from PIL import Image
import shutil

# # Remember to remove work/dataset and create a new empty dir before running.
# # Definition of the delete paths.
# del_path_dataset = 'work/dataset'
# del_path_training = 'work/trainImages'
# del_path_validation = 'work/evalImages'

# # Delete the dataset, training, and validation directories if any of them exsits.
# shutil.rmtree(del_path_dataset, ignore_errors=True)
# shutil.rmtree(del_path_training,ignore_errors=True)
# shutil.rmtree(del_path_validation,ignore_errors=True)

# Load some of the training and validation images for display.
imgs = []
paths = os.listdir('work/dataset') # Path of the training and validation image.
for path in paths:   
    img_path = os.path.join('work/dataset', path)
    # print(img_path)
    if os.path.isdir(img_path):
        img_paths = os.listdir(img_path)
        img_random_choice = random.choice(img_paths) # Get randomly choosed images' file name.
        img = Image.open(os.path.join(img_path, img_random_choice))
        imgs.append((img, path)) # Reading images from the folder.

# Display some random images from the dataset.
f, ax = plt.subplots(4, 3, figsize=(12,12)) # Resize each axes to the same as each other.
for i, img in enumerate(imgs[:12]):
    ax[i//3, i%3].imshow(img[0])
    ax[i//3, i%3].axis('off')
    ax[i//3, i%3].set_title('label: %s' % img[1])
plt.show() # Show the example random choosed 9 images.

# Split the dataset into training and validation sets.
!python NonLocalNet_Codes/train_val_split.py

# Network training for Non-Local Net with global contextual information extraction ability.
!python NonLocalNet_Codes/train.py --net 'gc'

# Network validation for Non-Local Net with global contextual information extraction ability.
!python NonLocalNet_Codes/eval.py --net 'gc'
