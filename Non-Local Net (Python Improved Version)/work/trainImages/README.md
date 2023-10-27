## IV. CORNER FEATURE MAP RECOGNITION BASED ON NON-LOCAL NETWORK

### A. Theory in Simple 

The Non-Local mechanism is a technique used to capture long-range dependencies in an image or feature map. It allows the network to consider non-local relationships between pixels or regions, which is particularly useful for tasks like corner feature detection. To implement corner feature map recognition using a CNN with Non-Local mechanism, we start by designing a network architecture that incorporates Non-Local blocks. These blocks are inserted into the network's layers to capture global information and enhance the network's ability for sparse feature extraction.

![GC-ResNeXt网络](https://github.com/JoeyBGOfficial/Through-the-Wall-Radar-Human-Activity-Recognition-Based-on-M-D-Corner-Feature-and-Non-Local-Net/assets/67720072/d4eb1212-d16d-4245-89a3-ed3536225d05)

Fig. 4. Details of the proposed Non-Local neural network's structure.

### B. Codes Explanation (Folder: Non-Local Net (Matlab Version), Non-Local Net (Python Improved Version))

#### 1. NonLocal_Net.m:

The script achieves the MATLAB version of whole process of Non-Local Net construction, network data input, network data preprocessing, real-time training and validation.

#### 2. NonLocalNet.py:

The script achieves the Python, Paddlepaddle framework improved version of whole process of Non-Local Net construction, network data input, network data preprocessing, real-time training and validation. The code also supports command-line one-click replacement of network module structure. Part of the codes come from the work of nanting03, and are improved by us. The supporting functions are all placed in the NonLocalNet_Codes folder, each as follows:

TABLE I. FUNCTIONS OF NON-LOCAL NET IN PYTHON VERSION
| Names of the codes | Functions |
| ------------------ | --------- |
| non_local.py | Definition of Non-Local Modules Part. 1 |
| context_block.py | Definition of Non-Local Modules Part. 2 |
|nlnet.py | Definition of the Network |
|radar_har_dataset.py | Definition of the Dataset |
| config.py | Configurations |
| train_val_split.py | Splitting the Training and Validation Sets |
| train.py | Start Model Training |
| eval.py | Start Model Validation |

### C. Datafiles Explanation (Folder: Non-Local Net (Matlab Version), Non-Local Net (Python Improved Version))

#### 1. Hyperparam.mat:

The .mat file defines the hyperparameters and relationships between the various layers of the MATLAB version of the network and contains some of the information needed for training.

