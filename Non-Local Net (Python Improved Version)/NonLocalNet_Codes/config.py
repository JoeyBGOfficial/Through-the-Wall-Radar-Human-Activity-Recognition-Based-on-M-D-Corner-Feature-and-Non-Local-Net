# Configurations of your Program
# Former Author: nanting03, JoeyBG;
# Improved By: JoeyBG;
# Affiliation: Beijing Institute of Technology, Radar Research Lab;
# Date: 2023-9-1;
# Language & Platform: Python, Paddlepaddle Framework.
#
# Introduction:
# Make sure the classification output, paths, epoches, batch size, and initial learning rate are set to the value you want.
#

# Configurations of your Program
config_parameters = {
    "class_dim": 3,  # Total number of classes, which can be changed.
    "target_path": " ", # Change it to the path you want to store your dataset and model.                   
    'train_image_dir': " ", # Change it to the path where your training images store.
    'eval_image_dir': " ", # Change it to the path where your evaluation images store.
    'epochs':20, # Total number of traing epoches, which can be changed.
    'batch_size': 64, # Mini-Batch size, which can also be changed (In our paper, it is set to 32).
    'lr': 0.005, # Initial Learning rate, which can also be changed (In our paper, it is set to 0.00147).
}