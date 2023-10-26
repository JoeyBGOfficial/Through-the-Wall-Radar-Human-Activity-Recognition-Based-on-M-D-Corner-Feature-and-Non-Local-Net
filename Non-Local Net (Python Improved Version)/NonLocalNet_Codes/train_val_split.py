# Configurations of your Program
# Former Author: nanting03, JoeyBG;
# Improved By: JoeyBG;
# Affiliation: Beijing Institute of Technology, Radar Research Lab;
# Date: 2023-9-1;
# Language & Platform: Python, Paddlepaddle Framework.
#
# Introduction:
# This is the code used for randomly splitting training and validation dataset.
#

import os
import shutil
from config import config_parameters

train_dir = config_parameters['train_image_dir']
eval_dir = config_parameters['eval_image_dir']
paths = os.listdir('work/dataset')

if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(eval_dir):
    os.mkdir(eval_dir)

for path in paths:
    imgs_dir = os.listdir(os.path.join('work/dataset', path))
    target_train_dir = os.path.join(train_dir,path)
    target_eval_dir = os.path.join(eval_dir,path)
    if not os.path.exists(target_train_dir):
        os.mkdir(target_train_dir)
    if not os.path.exists(target_eval_dir):
        os.mkdir(target_eval_dir)
    for i in range(len(imgs_dir)):
        if ' ' in imgs_dir[i]:
            new_name = imgs_dir[i].replace(' ', '_')
        else:
            new_name = imgs_dir[i]
        target_train_path = os.path.join(target_train_dir, new_name)
        target_eval_path = os.path.join(target_eval_dir, new_name)     
        if i % 5 == 0:
            shutil.copyfile(os.path.join(os.path.join('work/dataset', path), imgs_dir[i]), target_eval_path)
        else:
            shutil.copyfile(os.path.join(os.path.join('work/dataset', path), imgs_dir[i]), target_train_path)

print('finished train val split!')
