# Configurations of your Program
# Former Author: nanting03, JoeyBG;
# Improved By: JoeyBG;
# Affiliation: Beijing Institute of Technology, Radar Research Lab;
# Date: 2023-9-1;
# Language & Platform: Python, Paddlepaddle Framework.
#
# Introduction:
# This is the common code used for basic dataloader definition in Paddlepaddle framework.
#

import paddle
import numpy as np
from typing import Callable
from config import config_parameters

class RADAR_HAR(paddle.io.Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self, transforms: Callable, mode: str ='train'):
        """
        步骤二：实现构造函数，定义数据读取方式
        """
        super(RADAR_HAR, self).__init__()
        
        self.mode = mode
        self.transforms = transforms

        train_image_dir = config_parameters['train_image_dir']
        eval_image_dir = config_parameters['eval_image_dir']

        train_data_folder = paddle.vision.DatasetFolder(train_image_dir)
        eval_data_folder = paddle.vision.DatasetFolder(eval_image_dir)
        
        if self.mode  == 'train':
            self.data = train_data_folder
        elif self.mode  == 'eval':
            self.data = eval_data_folder


    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        data = np.array(self.data[index][0]).astype('float32')

        data = self.transforms(data)

        label = np.array([self.data[index][1]]).astype('int64')
        
        return data, label

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.data)