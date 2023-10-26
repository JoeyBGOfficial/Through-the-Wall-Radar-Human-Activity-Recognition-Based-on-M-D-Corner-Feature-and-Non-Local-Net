# Configurations of your Program
# Former Author: nanting03, JoeyBG;
# Improved By: JoeyBG;
# Affiliation: Beijing Institute of Technology, Radar Research Lab;
# Date: 2023-9-1;
# Language & Platform: Python, Paddlepaddle Framework.
#
# Introduction:
# This is the code used for running the model for training.
# Three selections of Non-Local mechanism are provided:
# 1. Traditional Non-Local Module: 'nl';
# 2. Non-Local Module with Global Contextual Information Extraction Ability: 'gc';
# 3. Non-Local Module with Transformer Attention Embedded: 'bat'.
#

# Import the necessary libraries.
from config import config_parameters
import paddle.vision.transforms as T
from radar_har_dataset import RADAR_HAR
from nlnet import nlnet50
import paddle
import argparse
import warnings

# Ignore the filter warnings.
warnings.filterwarnings("ignore")

# Load a parser for training preparations.
parser = argparse.ArgumentParser(description='oct_resnet Training')

# Parser construction.
parser.add_argument('--net', default='no',  type=str,
                    help='the arch to use')
parser.add_argument('--num_classes', default=config_parameters['class_dim'], type=int,
                    help='number of classes for classification')
parser.add_argument('--weights', default='no',  type=str,
                    help='the path for pretrained model')
parser.add_argument('--pretrained', default=False,  type=bool,
                    help='whether to load pretrained weights')
parser.add_argument('--batch_size', default=config_parameters['batch_size'],  type=int,
                    help='batch_size')
parser.add_argument('--lr', default=config_parameters['lr'], type=float)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--epochs', type = int, default = config_parameters['epochs'])
parser.add_argument('--warmup', type = int, default = 10)
args = parser.parse_args()

# Training definitions for data preprocessing.
train_transforms = T.Compose([
            T.Resize(224), 
            # T.CenterCrop(224),
            # T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        ])

# Validation definitions for data preprocessing.
eval_transforms = T.Compose([
    T.Resize(224), 
    # T.CenterCrop(224),
    # T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
])

# Define training variable for Network.
Radar_HAR_train = RADAR_HAR(transforms=train_transforms, mode='train')
Radar_HAR_eval = RADAR_HAR(transforms=eval_transforms, mode='eval')

# Training definitions for data batch loader.
train_loader = paddle.io.DataLoader(Radar_HAR_train, places=paddle.CUDAPlace(0), batch_size=args.batch_size, shuffle=True)

# Evaluation definitions for data batch loader.
eval_loader = paddle.io.DataLoader(Radar_HAR_eval, places=paddle.CUDAPlace(0), batch_size= 64)

# Selections for different models.
if args.net == 'nl':
    model = nlnet50(num_classes=args.num_classes, nltype='nl')
if args.net == 'bat':
    model = nlnet50(num_classes=args.num_classes, nltype='bat')
if args.net == 'gc':
    model = nlnet50(num_classes=args.num_classes, nltype='gc')
elif args.net == 'resnet':
    model = paddle.vision.models.resnet50(num_classes=args.num_classes)

# Loading pretrained models if necessary.
if args.pretrained:
    weights = paddle.load(args.weights)
    model.set_state_dict(weights)
    print('loading pretrained models')

# Define the class variable for best model saving.
class SaveBestModel(paddle.callbacks.Callback):
    def __init__(self, target=0.5, path='./best_model', verbose=0):
        self.target = target
        self.epoch = None
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch

    def on_eval_end(self, logs=None):
        if logs.get('acc') > self.target:
            self.target = logs.get('acc')
            self.model.save(self.path)
            print('best acc is {} at epoch {}'.format(self.target, self.epoch))

# Call back for VisualDL.
callback_visualdl = paddle.callbacks.VisualDL(log_dir=args.net)
callback_savebestmodel = SaveBestModel(target=0.5, path=args.net)
callbacks = [callback_visualdl, callback_savebestmodel]

# Parameters for learning.
base_lr = args.lr
wamup_steps = args.warmup
epochs = args.epochs

# Define the optimizers: momentum SGD, learning rate dropping: 0.9.
def make_optimizer(parameters=None):
    momentum = 0.9
    learning_rate= paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=base_lr, T_max=epochs, verbose=False)
    learning_rate = paddle.optimizer.lr.LinearWarmup(
        learning_rate=learning_rate,
        warmup_steps=wamup_steps,
        start_lr=base_lr / 5.,
        end_lr=base_lr,
        verbose=False)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=momentum,
        parameters=parameters)
    return optimizer

# Compile the optimizer.
optimizer = make_optimizer(model.parameters())

# Define the model for training.
model = paddle.Model(model)

# Model preparation.
model.prepare(optimizer,
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy()) 

# Official model training.
model.fit(train_loader,
          eval_loader,
          epochs=epochs,
          callbacks=callbacks,
          verbose=1)