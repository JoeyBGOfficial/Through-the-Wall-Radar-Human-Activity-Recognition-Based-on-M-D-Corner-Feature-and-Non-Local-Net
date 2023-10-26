# Configurations of your Program
# Former Author: nanting03, JoeyBG;
# Improved By: JoeyBG;
# Affiliation: Beijing Institute of Technology, Radar Research Lab;
# Date: 2023-9-1;
# Language & Platform: Python, Paddlepaddle Framework.
#
# Introduction:
# This is the code used for running the model for evaluation (Both validation and testing can use this code).
#

from config import config_parameters
import paddle.vision.transforms as T
from radar_har_dataset import RADAR_HAR
from nlnet import nlnet50
import paddle
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='oct_resnet evaluating')

parser.add_argument('--net', default='no',  type=str,
                    help='the arch to use')
parser.add_argument('--num_classes', default=config_parameters['class_dim'], type=int,
                    help='number of classes for classification')
args = parser.parse_args()

eval_transforms = T.Compose([
    T.Resize(224), 
    # T.CenterCrop(224),
    # T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
])

Radar_HAR_eval = RADAR_HAR(transforms=eval_transforms, mode='eval')

eval_loader = paddle.io.DataLoader(Radar_HAR_eval, places=paddle.CUDAPlace(0), batch_size= 32)

if args.net == 'nl':
    model = nlnet50(num_classes=args.num_classes, nltype='nl')
if args.net == 'bat':
    model = nlnet50(num_classes=args.num_classes, nltype='bat')
if args.net == 'gc':
    model = nlnet50(num_classes=args.num_classes, nltype='gc')
elif args.net == 'resnet':
    model = paddle.vision.models.resnet50(num_classes=args.num_classes)
        
weights = paddle.load(args.net+'.pdparams')
model.set_state_dict(weights)

model = paddle.Model(model)

model.prepare(loss=paddle.nn.CrossEntropyLoss(),
            metrics=paddle.metric.Accuracy())

result = model.evaluate(eval_loader, verbose=1)
print(result)