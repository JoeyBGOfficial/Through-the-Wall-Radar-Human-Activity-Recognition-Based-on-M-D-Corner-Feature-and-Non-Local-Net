## V. NUMERICAL SIMULATIONS AND EXPERIMENTS

### A. Theory in Simple 

In the paper we focus on the visualization, comparison and ablation validation of the performance of the proposed method. The visualization contains feature maps, training process. Comparison contains accuracy, generalization, robustness, etc. The ablation addresses the design necessity of each module.

![实验场景示意图](https://github.com/JoeyBGOfficial/Through-the-Wall-Radar-Human-Activity-Recognition-Based-on-M-D-Corner-Feature-and-Non-Local-Net/assets/67720072/8f7cdc04-34e9-4cf0-9331-79413eb3b0b0)

Fig. 5. Schematic diagram of the simulation and experimental scenes.

### B. Codes Explanation (Folder: Confusion Matrices, SequenceNN, T-SNE Tools)

#### 1. Confusion_Matrix_Generator.m:

The code implements the input of a confusion matrix in matrix form and outputs the result of its visualization, with the option of a slower but aesthetically pleasing generation or a faster but simplified one.

#### 2. SequenceNN.m:

Here we provide a construction scheme that combines ResNet with sequential neural networks as a comparative reference of sorts. The method is not given in the paper, but is still valid.

#### 3. T_SNE.m:

Input multiple sets of images of different categories and output their scatterplots reduced to three dimensions by T-SNE algorithm.

### C. Datafiles Explanation (Folder: Confusion Matrices, SequenceNN, T-SNE Tools)

#### 1. Hyperparam.mat:

The .mat file defines the hyperparameters and relationships between the various layers of the MATLAB version of the sequential network and contains some of the information needed for training.

#### 2. TSNE_Clist.mat:

This data file stores the color maps used to generate T-SNE scatter Plots, which is a tool that corresponds to the color maps in the paper.

