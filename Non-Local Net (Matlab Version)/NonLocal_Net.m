%% Create and Train Non-Local Neural Network Model for Recognition
% Former Author: JoeyBG;
% Improved By: JoeyBG;
% Affiliation: Beijing Institute of Technology, Radar Research Lab;
% Date: 2023-9-1;
% Language & Platform: MATLAB R2023a.
%
% Introduction:
% The complete structure of the proposed Non-Local Net can be divided into
% two parts: The improved ResNeXt-based backbone and Non-Local module.
%
% ResNeXt is a deep residual neural network architecture that extends and 
% improves upon the ResNet (Residual Network). The design philosophy of 
% ResNeXt aims to enhance the model's representation and feature extraction 
% capabilities.
% Residual Connections: 
% ResNeXt builds upon the core idea of ResNet, which is to address the 
% vanishing gradient and training difficulty by utilizing skip connections 
% or residual connections across layers. These connections allow the 
% information to propagate directly through the network, facilitating 
% better learning and preservation of important features.
% Grouped Convolutions: ResNeXt introduces the concept of grouped convolutions 
% to increase the model's width and representation capacity. Traditional 
% convolutional operations operate across all input channels, whereas grouped 
% convolutions divide the input channels into smaller groups and perform 
% independent convolutions within each group. This grouping reduces the 
% computational complexity to some extent and allows increasing the width 
% of the network by increasing the number of groups, thus enhancing the 
% model's expressive power.
% The network we propose is a simplified version of ResNeXt, 
% consisting of 16-channel grouped convolutions with 4 levels of depth for 
% feature extraction, along with a multilayer perceptron for decision-making.
%
% The Non-Local mechanism in neural networks is a technique that allows 
% capturing long-range dependencies and modeling relationships between 
% distant spatial or temporal locations in data. It is used to enhance the 
% capability of neural networks in capturing global context and improving 
% performance in tasks such as image recognition, video understanding, and 
% natural language processing.
% The Non-Local mechanism introduces non-local operations into neural 
% network architectures, which enable each position in the input to attend 
% to all other positions, regardless of their spatial or temporal distance. 
% This is achieved by computing the affinity between different positions 
% and using it to weight the contribution of each position to the final output.
% The key idea behind the Non-Local mechanism is that distant positions 
% can provide valuable information for understanding a particular position. 
% By allowing interactions between all positions, the network can capture 
% long-range dependencies and capture global context, leading to better 
% understanding and representation of the data.
% The Non-Local mechanism we use here is a method for extracting global 
% contextual information. It allows the network to capture long-range 
% dependencies and model relationships between distant positions in the 
% data, enabling a comprehensive understanding of the global context. 
% By incorporating the Non-Local mechanism, our network can effectively 
% extract and utilize global contextual information to enhance its 
% performance in the given task.
%
% Script for creating and training a deep learning network with the 
% following properties:
% Number of layers: 503;
% Number of connections: 630;
% Training setup file: Non-LocalNetSourceFiles\Hyperparam.mat.
%
% Run this script to create the network layers, import training and 
% validation data, and train the network. 
% The network layers are stored in the workspace variable lgraph.
% The trained network is stored in the workspace variable net.

%% Initialization of Matlab Script
clear all;
close all;
clc;

%% Load Initial Parameters
% Load parameters for network initialization. 
% For transfer learning, the network initialization parameters are the parameters of the initial pretrained network.
trainingSetup = load("Non-LocalNetSourceFiles\Hyperparam.mat");

%% Import Data
% Import training and validation data.
% Read data from file: Corner Dataset/ as an example.
% The function imageDatastore automatically divides and labels data sets by file name.
Dataset_Path = " "; % Change the data path to your own dataset.
if Dataset_Path == " "
    error("Dataset_Path is empty! Change it to your own path where dataset stores.")
end
imdsTrain = imageDatastore(Dataset_Path,"IncludeSubfolders",true,"LabelSource","foldernames"); % Inport dataset.
[imdsTrain, imdsValidation] = splitEachLabel(imdsTrain,0.8,"randomized"); % The number of training data:the number of validation data = 4:1.

% Resize the images to match the network input layer.
% The input image should be in 6 channels (3 for R2TM and 3 for D2TM, 
% concatenated in channel direction), but whatever spatial size you want.
augimdsTrain = augmentedImageDatastore([256 256 6],imdsTrain);
augimdsValidation = augmentedImageDatastore([256 256 6],imdsValidation);

%% Set Training Options
% Specify options to use when training.
% The following are just the suggestion of the training options:
% Optimizer: Adam.
% Execution environment: GPU or CPU, choose automatically.
% Initial learning rate: 0.00147.
% Training epoches: 20.
% Batch size: 32.
% Shuffle frequency: per epoch.
% Validation frequency: per 10 batches.
% Plot: training progress.
% Validation data: from the variable "augimdsValidation" defined above.
opts = trainingOptions("adam",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.00147,...
    "MaxEpochs",20,...
    "MiniBatchSize",32,...
    "Shuffle","every-epoch",...
    "ValidationFrequency",10,...
    "Plots","training-progress",...
    "ValidationData",augimdsValidation);

%% Create Layer Graph
% Create the layer graph variable to contain the network layers.
lgraph = layerGraph();

%% Add Layer Branches
% Add the branches of the network to the layer graph. Each branch is a linear array of layers.
% These are the layers with convolution operations and residual links.
tempLayers = [
    imageInputLayer([256 256 6],"Name","imageinput")
    convolution2dLayer([1 1],8,"Name","conv","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_1","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_17","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_33","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_2","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_18","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_34","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_3","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_19","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_35","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_4","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_20","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_36","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_5","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_21","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_37","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_6","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_22","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_38","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_7","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_23","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_39","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_8","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_24","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_40","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_9","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_25","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_41","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_10","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_26","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_42","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_11","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_27","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_43","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_12","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_28","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_44","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_13","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_29","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_45","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_14","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_30","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_46","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_15","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_31","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_47","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_16","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_32","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_48","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(16,"Name","addition")
    layerNormalizationLayer("Name","layernorm")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_1")
    reluLayer("Name","relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_49","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 1])
    softmaxLayer("Name","softmax")
    convolution2dLayer([1 1],1,"Name","conv_53","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_1","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])
    convolution2dLayer([1 1],8,"Name","conv_52","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication")
    resize3dLayer("Name","resize3d-output-size_8","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_54","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_2","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[64 64 1])
    softmaxLayer("Name","softmax_2")
    convolution2dLayer([1 1],1,"Name","conv_56","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_3","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[64 64 8])
    convolution2dLayer([1 1],8,"Name","conv_55","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_2")
    resize3dLayer("Name","resize3d-output-size_9","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_57","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_4","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[16 16 1])
    softmaxLayer("Name","softmax_3")
    convolution2dLayer([1 1],1,"Name","conv_59","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_5","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[16 16 8])
    convolution2dLayer([1 1],8,"Name","conv_58","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_3")
    resize3dLayer("Name","resize3d-output-size_10","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_60","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_6","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[4 4 1])
    softmaxLayer("Name","softmax_4")
    convolution2dLayer([1 1],1,"Name","conv_62","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_7","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[4 4 8])
    convolution2dLayer([1 1],8,"Name","conv_61","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_4")
    resize3dLayer("Name","resize3d-output-size_11","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_63","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_12","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[128 128 1])
    softmaxLayer("Name","softmax_5")
    convolution2dLayer([1 1],1,"Name","conv_65","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_13","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[128 128 8])
    convolution2dLayer([1 1],8,"Name","conv_64","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_5")
    resize3dLayer("Name","resize3d-output-size_14","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_66","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_15","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[32 32 1])
    softmaxLayer("Name","softmax_6")
    convolution2dLayer([1 1],1,"Name","conv_68","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_16","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[32 32 8])
    convolution2dLayer([1 1],8,"Name","conv_67","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_6")
    resize3dLayer("Name","resize3d-output-size_17","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_69","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_18","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[8 8 1])
    softmaxLayer("Name","softmax_7")
    convolution2dLayer([1 1],1,"Name","conv_71","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_19","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[8 8 8])
    convolution2dLayer([1 1],8,"Name","conv_70","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_7")
    resize3dLayer("Name","resize3d-output-size_20","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_72","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_21","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[2 2 1])
    softmaxLayer("Name","softmax_8")
    convolution2dLayer([1 1],1,"Name","conv_74","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_22","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[2 2 8])
    convolution2dLayer([1 1],8,"Name","conv_73","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_8")
    resize3dLayer("Name","resize3d-output-size_23","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(8,"Name","addition_2")
    globalAveragePooling2dLayer("Name","gapool")
    convolution2dLayer([1 1],4,"Name","conv_50","Padding","same")
    layerNormalizationLayer("Name","layernorm_1")
    reluLayer("Name","relu_1")
    convolution2dLayer([1 1],8,"Name","conv_51","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_1")
    convolution2dLayer([1 1],8,"Name","conv_75","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_76","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_92","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_108","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_77","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_93","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_109","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_78","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_94","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_110","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_79","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_95","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_111","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_80","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_96","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_112","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_81","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_97","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_113","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_82","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_98","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_114","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_83","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_99","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_115","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_84","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_100","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_116","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_85","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_101","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_117","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_86","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_102","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_118","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_87","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_103","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_119","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_88","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_104","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_120","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_89","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_105","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_121","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_90","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_106","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_122","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_91","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_107","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_123","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(16,"Name","addition_3")
    layerNormalizationLayer("Name","layernorm_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_4")
    reluLayer("Name","relu_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_124","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_24","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 1])
    softmaxLayer("Name","softmax_9")
    convolution2dLayer([1 1],1,"Name","conv_128","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_25","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])
    convolution2dLayer([1 1],8,"Name","conv_127","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_9")
    resize3dLayer("Name","resize3d-output-size_32","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_129","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_26","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[64 64 1])
    softmaxLayer("Name","softmax_10")
    convolution2dLayer([1 1],1,"Name","conv_131","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_27","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[64 64 8])
    convolution2dLayer([1 1],8,"Name","conv_130","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_11")
    resize3dLayer("Name","resize3d-output-size_33","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_132","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_28","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[16 16 1])
    softmaxLayer("Name","softmax_11")
    convolution2dLayer([1 1],1,"Name","conv_134","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_29","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[16 16 8])
    convolution2dLayer([1 1],8,"Name","conv_133","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_12")
    resize3dLayer("Name","resize3d-output-size_34","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_135","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_30","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[4 4 1])
    softmaxLayer("Name","softmax_12")
    convolution2dLayer([1 1],1,"Name","conv_137","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_31","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[4 4 8])
    convolution2dLayer([1 1],8,"Name","conv_136","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_13")
    resize3dLayer("Name","resize3d-output-size_35","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_138","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_36","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[128 128 1])
    softmaxLayer("Name","softmax_13")
    convolution2dLayer([1 1],1,"Name","conv_140","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_37","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[128 128 8])
    convolution2dLayer([1 1],8,"Name","conv_139","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_14")
    resize3dLayer("Name","resize3d-output-size_38","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_141","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_39","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[32 32 1])
    softmaxLayer("Name","softmax_14")
    convolution2dLayer([1 1],1,"Name","conv_143","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_40","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[32 32 8])
    convolution2dLayer([1 1],8,"Name","conv_142","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_15")
    resize3dLayer("Name","resize3d-output-size_41","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_144","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_42","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[8 8 1])
    softmaxLayer("Name","softmax_15")
    convolution2dLayer([1 1],1,"Name","conv_146","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_43","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[8 8 8])
    convolution2dLayer([1 1],8,"Name","conv_145","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_16")
    resize3dLayer("Name","resize3d-output-size_44","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_147","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_45","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[2 2 1])
    softmaxLayer("Name","softmax_16")
    convolution2dLayer([1 1],1,"Name","conv_149","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_46","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[2 2 8])
    convolution2dLayer([1 1],8,"Name","conv_148","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_17")
    resize3dLayer("Name","resize3d-output-size_47","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(8,"Name","addition_5")
    globalAveragePooling2dLayer("Name","gapool_1")
    convolution2dLayer([1 1],4,"Name","conv_125","Padding","same")
    layerNormalizationLayer("Name","layernorm_3")
    reluLayer("Name","relu_3")
    convolution2dLayer([1 1],8,"Name","conv_126","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_10")
    convolution2dLayer([1 1],8,"Name","conv_150","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_151","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_167","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_183","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_152","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_168","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_184","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_153","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_169","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_185","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_154","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_170","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_186","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_155","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_171","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_187","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_156","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_172","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_188","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_157","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_173","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_189","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_158","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_174","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_190","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_159","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_175","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_191","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_160","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_176","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_192","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_161","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_177","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_193","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_162","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_178","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_194","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_163","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_179","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_195","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_164","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_180","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_196","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_165","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_181","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_197","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_166","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_182","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_198","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(16,"Name","addition_6")
    layerNormalizationLayer("Name","layernorm_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_7")
    reluLayer("Name","relu_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_199","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_48","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 1])
    softmaxLayer("Name","softmax_17")
    convolution2dLayer([1 1],1,"Name","conv_203","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_49","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])
    convolution2dLayer([1 1],8,"Name","conv_202","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_18")
    resize3dLayer("Name","resize3d-output-size_56","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_204","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_50","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[64 64 1])
    softmaxLayer("Name","softmax_18")
    convolution2dLayer([1 1],1,"Name","conv_206","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_51","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[64 64 8])
    convolution2dLayer([1 1],8,"Name","conv_205","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_20")
    resize3dLayer("Name","resize3d-output-size_57","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_207","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_52","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[16 16 1])
    softmaxLayer("Name","softmax_19")
    convolution2dLayer([1 1],1,"Name","conv_209","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_53","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[16 16 8])
    convolution2dLayer([1 1],8,"Name","conv_208","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_21")
    resize3dLayer("Name","resize3d-output-size_58","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_210","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_54","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[4 4 1])
    softmaxLayer("Name","softmax_20")
    convolution2dLayer([1 1],1,"Name","conv_212","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_55","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[4 4 8])
    convolution2dLayer([1 1],8,"Name","conv_211","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_22")
    resize3dLayer("Name","resize3d-output-size_59","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_213","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_60","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[128 128 1])
    softmaxLayer("Name","softmax_21")
    convolution2dLayer([1 1],1,"Name","conv_215","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_61","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[128 128 8])
    convolution2dLayer([1 1],8,"Name","conv_214","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_23")
    resize3dLayer("Name","resize3d-output-size_62","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_216","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_63","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[32 32 1])
    softmaxLayer("Name","softmax_22")
    convolution2dLayer([1 1],1,"Name","conv_218","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_64","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[32 32 8])
    convolution2dLayer([1 1],8,"Name","conv_217","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_24")
    resize3dLayer("Name","resize3d-output-size_65","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_219","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_66","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[8 8 1])
    softmaxLayer("Name","softmax_23")
    convolution2dLayer([1 1],1,"Name","conv_221","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_67","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[8 8 8])
    convolution2dLayer([1 1],8,"Name","conv_220","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_25")
    resize3dLayer("Name","resize3d-output-size_68","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_222","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_69","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[2 2 1])
    softmaxLayer("Name","softmax_24")
    convolution2dLayer([1 1],1,"Name","conv_224","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_70","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[2 2 8])
    convolution2dLayer([1 1],8,"Name","conv_223","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_26")
    resize3dLayer("Name","resize3d-output-size_71","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(8,"Name","addition_8")
    globalAveragePooling2dLayer("Name","gapool_2")
    convolution2dLayer([1 1],4,"Name","conv_200","Padding","same")
    layerNormalizationLayer("Name","layernorm_5")
    reluLayer("Name","relu_5")
    convolution2dLayer([1 1],8,"Name","conv_201","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_19")
    convolution2dLayer([1 1],8,"Name","conv_225","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_226","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_242","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_258","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_227","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_243","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_259","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_228","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_244","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_260","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_229","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_245","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_261","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_230","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_246","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_262","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_231","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_247","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_263","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_232","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_248","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_264","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_233","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_249","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_265","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_234","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_250","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_266","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_235","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_251","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_267","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_236","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_252","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_268","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_237","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_253","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_269","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_238","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_254","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_270","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_239","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_255","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_271","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_240","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_256","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_272","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv_241","Padding","same")
    convolution2dLayer([3 3],4,"Name","conv_257","Padding","same")
    convolution2dLayer([1 1],8,"Name","conv_273","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(16,"Name","addition_9")
    layerNormalizationLayer("Name","layernorm_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_10")
    reluLayer("Name","relu_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_274","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_72","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 1])
    softmaxLayer("Name","softmax_25")
    convolution2dLayer([1 1],1,"Name","conv_278","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_73","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])
    convolution2dLayer([1 1],8,"Name","conv_277","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_27")
    resize3dLayer("Name","resize3d-output-size_80","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_279","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_74","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[64 64 1])
    softmaxLayer("Name","softmax_26")
    convolution2dLayer([1 1],1,"Name","conv_281","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_75","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[64 64 8])
    convolution2dLayer([1 1],8,"Name","conv_280","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_29")
    resize3dLayer("Name","resize3d-output-size_81","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_282","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_76","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[16 16 1])
    softmaxLayer("Name","softmax_27")
    convolution2dLayer([1 1],1,"Name","conv_284","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_77","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[16 16 8])
    convolution2dLayer([1 1],8,"Name","conv_283","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_30")
    resize3dLayer("Name","resize3d-output-size_82","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_285","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_78","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[4 4 1])
    softmaxLayer("Name","softmax_28")
    convolution2dLayer([1 1],1,"Name","conv_287","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_79","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[4 4 8])
    convolution2dLayer([1 1],8,"Name","conv_286","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_31")
    resize3dLayer("Name","resize3d-output-size_83","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_288","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_84","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[128 128 1])
    softmaxLayer("Name","softmax_29")
    convolution2dLayer([1 1],1,"Name","conv_290","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_85","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[128 128 8])
    convolution2dLayer([1 1],8,"Name","conv_289","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_32")
    resize3dLayer("Name","resize3d-output-size_86","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_291","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_87","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[32 32 1])
    softmaxLayer("Name","softmax_30")
    convolution2dLayer([1 1],1,"Name","conv_293","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_88","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[32 32 8])
    convolution2dLayer([1 1],8,"Name","conv_292","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_33")
    resize3dLayer("Name","resize3d-output-size_89","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_294","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_90","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[8 8 1])
    softmaxLayer("Name","softmax_31")
    convolution2dLayer([1 1],1,"Name","conv_296","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_91","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[8 8 8])
    convolution2dLayer([1 1],8,"Name","conv_295","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_34")
    resize3dLayer("Name","resize3d-output-size_92","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],1,"Name","conv_297","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_93","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[2 2 1])
    softmaxLayer("Name","softmax_32")
    convolution2dLayer([1 1],1,"Name","conv_299","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    resize3dLayer("Name","resize3d-output-size_94","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[2 2 8])
    convolution2dLayer([1 1],8,"Name","conv_298","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_35")
    resize3dLayer("Name","resize3d-output-size_95","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[256 256 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(8,"Name","addition_11")
    globalAveragePooling2dLayer("Name","gapool_3")
    convolution2dLayer([1 1],4,"Name","conv_275","Padding","same")
    layerNormalizationLayer("Name","layernorm_7")
    reluLayer("Name","relu_7")
    convolution2dLayer([1 1],8,"Name","conv_276","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_28")
    flattenLayer("Name","flatten")
    fullyConnectedLayer(128,"Name","fc")
    fullyConnectedLayer(64,"Name","fc_1")
    fullyConnectedLayer(3,"Name","fc_2")
    softmaxLayer("Name","softmax_1")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% Clean up helper variable.
clear tempLayers;

%% Connect Layer Branches
% Connect all the branches of the network to create the network graph.
% Add all the layer variables defined above to the lgraph() container in the order in which they will be processed by the network.
lgraph = connectLayers(lgraph,"conv","conv_1");
lgraph = connectLayers(lgraph,"conv","conv_2");
lgraph = connectLayers(lgraph,"conv","conv_3");
lgraph = connectLayers(lgraph,"conv","conv_4");
lgraph = connectLayers(lgraph,"conv","conv_5");
lgraph = connectLayers(lgraph,"conv","conv_6");
lgraph = connectLayers(lgraph,"conv","conv_7");
lgraph = connectLayers(lgraph,"conv","conv_8");
lgraph = connectLayers(lgraph,"conv","conv_9");
lgraph = connectLayers(lgraph,"conv","conv_10");
lgraph = connectLayers(lgraph,"conv","conv_11");
lgraph = connectLayers(lgraph,"conv","conv_12");
lgraph = connectLayers(lgraph,"conv","conv_13");
lgraph = connectLayers(lgraph,"conv","conv_14");
lgraph = connectLayers(lgraph,"conv","conv_15");
lgraph = connectLayers(lgraph,"conv","conv_16");
lgraph = connectLayers(lgraph,"conv","addition_1/in2");
lgraph = connectLayers(lgraph,"conv_33","addition/in1");
lgraph = connectLayers(lgraph,"conv_34","addition/in2");
lgraph = connectLayers(lgraph,"conv_35","addition/in3");
lgraph = connectLayers(lgraph,"conv_36","addition/in4");
lgraph = connectLayers(lgraph,"conv_37","addition/in5");
lgraph = connectLayers(lgraph,"conv_38","addition/in6");
lgraph = connectLayers(lgraph,"conv_39","addition/in7");
lgraph = connectLayers(lgraph,"conv_40","addition/in12");
lgraph = connectLayers(lgraph,"conv_41","addition/in10");
lgraph = connectLayers(lgraph,"conv_42","addition/in9");
lgraph = connectLayers(lgraph,"conv_43","addition/in8");
lgraph = connectLayers(lgraph,"conv_44","addition/in11");
lgraph = connectLayers(lgraph,"conv_45","addition/in14");
lgraph = connectLayers(lgraph,"conv_46","addition/in15");
lgraph = connectLayers(lgraph,"conv_47","addition/in16");
lgraph = connectLayers(lgraph,"conv_48","addition/in13");
lgraph = connectLayers(lgraph,"layernorm","addition_1/in1");
lgraph = connectLayers(lgraph,"relu","conv_49");
lgraph = connectLayers(lgraph,"relu","conv_54");
lgraph = connectLayers(lgraph,"relu","conv_57");
lgraph = connectLayers(lgraph,"relu","conv_60");
lgraph = connectLayers(lgraph,"relu","conv_63");
lgraph = connectLayers(lgraph,"relu","conv_66");
lgraph = connectLayers(lgraph,"relu","conv_69");
lgraph = connectLayers(lgraph,"relu","conv_72");
lgraph = connectLayers(lgraph,"relu","multiplication_1/in2");
lgraph = connectLayers(lgraph,"conv_49","resize3d-output-size");
lgraph = connectLayers(lgraph,"conv_49","resize3d-output-size_1");
lgraph = connectLayers(lgraph,"conv_53","multiplication/in1");
lgraph = connectLayers(lgraph,"conv_52","multiplication/in2");
lgraph = connectLayers(lgraph,"conv_54","resize3d-output-size_2");
lgraph = connectLayers(lgraph,"conv_54","resize3d-output-size_3");
lgraph = connectLayers(lgraph,"conv_56","multiplication_2/in1");
lgraph = connectLayers(lgraph,"conv_55","multiplication_2/in2");
lgraph = connectLayers(lgraph,"conv_57","resize3d-output-size_4");
lgraph = connectLayers(lgraph,"conv_57","resize3d-output-size_5");
lgraph = connectLayers(lgraph,"conv_59","multiplication_3/in1");
lgraph = connectLayers(lgraph,"conv_58","multiplication_3/in2");
lgraph = connectLayers(lgraph,"conv_60","resize3d-output-size_6");
lgraph = connectLayers(lgraph,"conv_60","resize3d-output-size_7");
lgraph = connectLayers(lgraph,"conv_62","multiplication_4/in1");
lgraph = connectLayers(lgraph,"conv_61","multiplication_4/in2");
lgraph = connectLayers(lgraph,"resize3d-output-size_8","addition_2/in1");
lgraph = connectLayers(lgraph,"resize3d-output-size_9","addition_2/in2");
lgraph = connectLayers(lgraph,"resize3d-output-size_10","addition_2/in3");
lgraph = connectLayers(lgraph,"resize3d-output-size_11","addition_2/in4");
lgraph = connectLayers(lgraph,"conv_63","resize3d-output-size_12");
lgraph = connectLayers(lgraph,"conv_63","resize3d-output-size_13");
lgraph = connectLayers(lgraph,"conv_65","multiplication_5/in1");
lgraph = connectLayers(lgraph,"conv_64","multiplication_5/in2");
lgraph = connectLayers(lgraph,"resize3d-output-size_14","addition_2/in5");
lgraph = connectLayers(lgraph,"conv_66","resize3d-output-size_15");
lgraph = connectLayers(lgraph,"conv_66","resize3d-output-size_16");
lgraph = connectLayers(lgraph,"conv_68","multiplication_6/in1");
lgraph = connectLayers(lgraph,"conv_67","multiplication_6/in2");
lgraph = connectLayers(lgraph,"resize3d-output-size_17","addition_2/in6");
lgraph = connectLayers(lgraph,"conv_69","resize3d-output-size_18");
lgraph = connectLayers(lgraph,"conv_69","resize3d-output-size_19");
lgraph = connectLayers(lgraph,"conv_71","multiplication_7/in1");
lgraph = connectLayers(lgraph,"conv_70","multiplication_7/in2");
lgraph = connectLayers(lgraph,"resize3d-output-size_20","addition_2/in7");
lgraph = connectLayers(lgraph,"conv_72","resize3d-output-size_21");
lgraph = connectLayers(lgraph,"conv_72","resize3d-output-size_22");
lgraph = connectLayers(lgraph,"conv_74","multiplication_8/in1");
lgraph = connectLayers(lgraph,"conv_73","multiplication_8/in2");
lgraph = connectLayers(lgraph,"resize3d-output-size_23","addition_2/in8");
lgraph = connectLayers(lgraph,"conv_51","multiplication_1/in1");
lgraph = connectLayers(lgraph,"conv_75","conv_76");
lgraph = connectLayers(lgraph,"conv_75","conv_77");
lgraph = connectLayers(lgraph,"conv_75","conv_78");
lgraph = connectLayers(lgraph,"conv_75","conv_79");
lgraph = connectLayers(lgraph,"conv_75","conv_80");
lgraph = connectLayers(lgraph,"conv_75","conv_81");
lgraph = connectLayers(lgraph,"conv_75","conv_82");
lgraph = connectLayers(lgraph,"conv_75","conv_83");
lgraph = connectLayers(lgraph,"conv_75","conv_84");
lgraph = connectLayers(lgraph,"conv_75","conv_85");
lgraph = connectLayers(lgraph,"conv_75","conv_86");
lgraph = connectLayers(lgraph,"conv_75","conv_87");
lgraph = connectLayers(lgraph,"conv_75","conv_88");
lgraph = connectLayers(lgraph,"conv_75","conv_89");
lgraph = connectLayers(lgraph,"conv_75","conv_90");
lgraph = connectLayers(lgraph,"conv_75","conv_91");
lgraph = connectLayers(lgraph,"conv_75","addition_4/in2");
lgraph = connectLayers(lgraph,"conv_108","addition_3/in1");
lgraph = connectLayers(lgraph,"conv_109","addition_3/in2");
lgraph = connectLayers(lgraph,"conv_110","addition_3/in3");
lgraph = connectLayers(lgraph,"conv_111","addition_3/in4");
lgraph = connectLayers(lgraph,"conv_112","addition_3/in5");
lgraph = connectLayers(lgraph,"conv_113","addition_3/in6");
lgraph = connectLayers(lgraph,"conv_114","addition_3/in7");
lgraph = connectLayers(lgraph,"conv_115","addition_3/in12");
lgraph = connectLayers(lgraph,"conv_116","addition_3/in10");
lgraph = connectLayers(lgraph,"conv_117","addition_3/in9");
lgraph = connectLayers(lgraph,"conv_118","addition_3/in8");
lgraph = connectLayers(lgraph,"conv_119","addition_3/in11");
lgraph = connectLayers(lgraph,"conv_120","addition_3/in14");
lgraph = connectLayers(lgraph,"conv_121","addition_3/in15");
lgraph = connectLayers(lgraph,"conv_122","addition_3/in16");
lgraph = connectLayers(lgraph,"conv_123","addition_3/in13");
lgraph = connectLayers(lgraph,"layernorm_2","addition_4/in1");
lgraph = connectLayers(lgraph,"relu_2","conv_124");
lgraph = connectLayers(lgraph,"relu_2","conv_129");
lgraph = connectLayers(lgraph,"relu_2","conv_132");
lgraph = connectLayers(lgraph,"relu_2","conv_135");
lgraph = connectLayers(lgraph,"relu_2","conv_138");
lgraph = connectLayers(lgraph,"relu_2","conv_141");
lgraph = connectLayers(lgraph,"relu_2","conv_144");
lgraph = connectLayers(lgraph,"relu_2","conv_147");
lgraph = connectLayers(lgraph,"relu_2","multiplication_10/in2");
lgraph = connectLayers(lgraph,"conv_124","resize3d-output-size_24");
lgraph = connectLayers(lgraph,"conv_124","resize3d-output-size_25");
lgraph = connectLayers(lgraph,"conv_128","multiplication_9/in1");
lgraph = connectLayers(lgraph,"conv_127","multiplication_9/in2");
lgraph = connectLayers(lgraph,"conv_129","resize3d-output-size_26");
lgraph = connectLayers(lgraph,"conv_129","resize3d-output-size_27");
lgraph = connectLayers(lgraph,"conv_131","multiplication_11/in1");
lgraph = connectLayers(lgraph,"conv_130","multiplication_11/in2");
lgraph = connectLayers(lgraph,"conv_132","resize3d-output-size_28");
lgraph = connectLayers(lgraph,"conv_132","resize3d-output-size_29");
lgraph = connectLayers(lgraph,"conv_134","multiplication_12/in1");
lgraph = connectLayers(lgraph,"conv_133","multiplication_12/in2");
lgraph = connectLayers(lgraph,"conv_135","resize3d-output-size_30");
lgraph = connectLayers(lgraph,"conv_135","resize3d-output-size_31");
lgraph = connectLayers(lgraph,"conv_137","multiplication_13/in1");
lgraph = connectLayers(lgraph,"conv_136","multiplication_13/in2");
lgraph = connectLayers(lgraph,"resize3d-output-size_32","addition_5/in1");
lgraph = connectLayers(lgraph,"resize3d-output-size_33","addition_5/in2");
lgraph = connectLayers(lgraph,"resize3d-output-size_34","addition_5/in3");
lgraph = connectLayers(lgraph,"resize3d-output-size_35","addition_5/in4");
lgraph = connectLayers(lgraph,"conv_138","resize3d-output-size_36");
lgraph = connectLayers(lgraph,"conv_138","resize3d-output-size_37");
lgraph = connectLayers(lgraph,"conv_140","multiplication_14/in1");
lgraph = connectLayers(lgraph,"conv_139","multiplication_14/in2");
lgraph = connectLayers(lgraph,"resize3d-output-size_38","addition_5/in5");
lgraph = connectLayers(lgraph,"conv_141","resize3d-output-size_39");
lgraph = connectLayers(lgraph,"conv_141","resize3d-output-size_40");
lgraph = connectLayers(lgraph,"conv_143","multiplication_15/in1");
lgraph = connectLayers(lgraph,"conv_142","multiplication_15/in2");
lgraph = connectLayers(lgraph,"resize3d-output-size_41","addition_5/in6");
lgraph = connectLayers(lgraph,"conv_144","resize3d-output-size_42");
lgraph = connectLayers(lgraph,"conv_144","resize3d-output-size_43");
lgraph = connectLayers(lgraph,"conv_146","multiplication_16/in1");
lgraph = connectLayers(lgraph,"conv_145","multiplication_16/in2");
lgraph = connectLayers(lgraph,"resize3d-output-size_44","addition_5/in7");
lgraph = connectLayers(lgraph,"conv_147","resize3d-output-size_45");
lgraph = connectLayers(lgraph,"conv_147","resize3d-output-size_46");
lgraph = connectLayers(lgraph,"conv_149","multiplication_17/in1");
lgraph = connectLayers(lgraph,"conv_148","multiplication_17/in2");
lgraph = connectLayers(lgraph,"resize3d-output-size_47","addition_5/in8");
lgraph = connectLayers(lgraph,"conv_126","multiplication_10/in1");
lgraph = connectLayers(lgraph,"conv_150","conv_151");
lgraph = connectLayers(lgraph,"conv_150","conv_152");
lgraph = connectLayers(lgraph,"conv_150","conv_153");
lgraph = connectLayers(lgraph,"conv_150","conv_154");
lgraph = connectLayers(lgraph,"conv_150","conv_155");
lgraph = connectLayers(lgraph,"conv_150","conv_156");
lgraph = connectLayers(lgraph,"conv_150","conv_157");
lgraph = connectLayers(lgraph,"conv_150","conv_158");
lgraph = connectLayers(lgraph,"conv_150","conv_159");
lgraph = connectLayers(lgraph,"conv_150","conv_160");
lgraph = connectLayers(lgraph,"conv_150","conv_161");
lgraph = connectLayers(lgraph,"conv_150","conv_162");
lgraph = connectLayers(lgraph,"conv_150","conv_163");
lgraph = connectLayers(lgraph,"conv_150","conv_164");
lgraph = connectLayers(lgraph,"conv_150","conv_165");
lgraph = connectLayers(lgraph,"conv_150","conv_166");
lgraph = connectLayers(lgraph,"conv_150","addition_7/in2");
lgraph = connectLayers(lgraph,"conv_183","addition_6/in1");
lgraph = connectLayers(lgraph,"conv_184","addition_6/in2");
lgraph = connectLayers(lgraph,"conv_185","addition_6/in3");
lgraph = connectLayers(lgraph,"conv_186","addition_6/in4");
lgraph = connectLayers(lgraph,"conv_187","addition_6/in5");
lgraph = connectLayers(lgraph,"conv_188","addition_6/in6");
lgraph = connectLayers(lgraph,"conv_189","addition_6/in7");
lgraph = connectLayers(lgraph,"conv_190","addition_6/in12");
lgraph = connectLayers(lgraph,"conv_191","addition_6/in10");
lgraph = connectLayers(lgraph,"conv_192","addition_6/in9");
lgraph = connectLayers(lgraph,"conv_193","addition_6/in8");
lgraph = connectLayers(lgraph,"conv_194","addition_6/in11");
lgraph = connectLayers(lgraph,"conv_195","addition_6/in14");
lgraph = connectLayers(lgraph,"conv_196","addition_6/in15");
lgraph = connectLayers(lgraph,"conv_197","addition_6/in16");
lgraph = connectLayers(lgraph,"conv_198","addition_6/in13");
lgraph = connectLayers(lgraph,"layernorm_4","addition_7/in1");
lgraph = connectLayers(lgraph,"relu_4","conv_199");
lgraph = connectLayers(lgraph,"relu_4","conv_204");
lgraph = connectLayers(lgraph,"relu_4","conv_207");
lgraph = connectLayers(lgraph,"relu_4","conv_210");
lgraph = connectLayers(lgraph,"relu_4","conv_213");
lgraph = connectLayers(lgraph,"relu_4","conv_216");
lgraph = connectLayers(lgraph,"relu_4","conv_219");
lgraph = connectLayers(lgraph,"relu_4","conv_222");
lgraph = connectLayers(lgraph,"relu_4","multiplication_19/in2");
lgraph = connectLayers(lgraph,"conv_199","resize3d-output-size_48");
lgraph = connectLayers(lgraph,"conv_199","resize3d-output-size_49");
lgraph = connectLayers(lgraph,"conv_203","multiplication_18/in1");
lgraph = connectLayers(lgraph,"conv_202","multiplication_18/in2");
lgraph = connectLayers(lgraph,"conv_204","resize3d-output-size_50");
lgraph = connectLayers(lgraph,"conv_204","resize3d-output-size_51");
lgraph = connectLayers(lgraph,"conv_206","multiplication_20/in1");
lgraph = connectLayers(lgraph,"conv_205","multiplication_20/in2");
lgraph = connectLayers(lgraph,"conv_207","resize3d-output-size_52");
lgraph = connectLayers(lgraph,"conv_207","resize3d-output-size_53");
lgraph = connectLayers(lgraph,"conv_209","multiplication_21/in1");
lgraph = connectLayers(lgraph,"conv_208","multiplication_21/in2");
lgraph = connectLayers(lgraph,"conv_210","resize3d-output-size_54");
lgraph = connectLayers(lgraph,"conv_210","resize3d-output-size_55");
lgraph = connectLayers(lgraph,"conv_212","multiplication_22/in1");
lgraph = connectLayers(lgraph,"conv_211","multiplication_22/in2");
lgraph = connectLayers(lgraph,"resize3d-output-size_56","addition_8/in1");
lgraph = connectLayers(lgraph,"resize3d-output-size_57","addition_8/in2");
lgraph = connectLayers(lgraph,"resize3d-output-size_58","addition_8/in3");
lgraph = connectLayers(lgraph,"resize3d-output-size_59","addition_8/in4");
lgraph = connectLayers(lgraph,"conv_213","resize3d-output-size_60");
lgraph = connectLayers(lgraph,"conv_213","resize3d-output-size_61");
lgraph = connectLayers(lgraph,"conv_215","multiplication_23/in1");
lgraph = connectLayers(lgraph,"conv_214","multiplication_23/in2");
lgraph = connectLayers(lgraph,"resize3d-output-size_62","addition_8/in5");
lgraph = connectLayers(lgraph,"conv_216","resize3d-output-size_63");
lgraph = connectLayers(lgraph,"conv_216","resize3d-output-size_64");
lgraph = connectLayers(lgraph,"conv_218","multiplication_24/in1");
lgraph = connectLayers(lgraph,"conv_217","multiplication_24/in2");
lgraph = connectLayers(lgraph,"resize3d-output-size_65","addition_8/in6");
lgraph = connectLayers(lgraph,"conv_219","resize3d-output-size_66");
lgraph = connectLayers(lgraph,"conv_219","resize3d-output-size_67");
lgraph = connectLayers(lgraph,"conv_221","multiplication_25/in1");
lgraph = connectLayers(lgraph,"conv_220","multiplication_25/in2");
lgraph = connectLayers(lgraph,"resize3d-output-size_68","addition_8/in7");
lgraph = connectLayers(lgraph,"conv_222","resize3d-output-size_69");
lgraph = connectLayers(lgraph,"conv_222","resize3d-output-size_70");
lgraph = connectLayers(lgraph,"conv_224","multiplication_26/in1");
lgraph = connectLayers(lgraph,"conv_223","multiplication_26/in2");
lgraph = connectLayers(lgraph,"resize3d-output-size_71","addition_8/in8");
lgraph = connectLayers(lgraph,"conv_201","multiplication_19/in1");
lgraph = connectLayers(lgraph,"conv_225","conv_226");
lgraph = connectLayers(lgraph,"conv_225","conv_227");
lgraph = connectLayers(lgraph,"conv_225","conv_228");
lgraph = connectLayers(lgraph,"conv_225","conv_229");
lgraph = connectLayers(lgraph,"conv_225","conv_230");
lgraph = connectLayers(lgraph,"conv_225","conv_231");
lgraph = connectLayers(lgraph,"conv_225","conv_232");
lgraph = connectLayers(lgraph,"conv_225","conv_233");
lgraph = connectLayers(lgraph,"conv_225","conv_234");
lgraph = connectLayers(lgraph,"conv_225","conv_235");
lgraph = connectLayers(lgraph,"conv_225","conv_236");
lgraph = connectLayers(lgraph,"conv_225","conv_237");
lgraph = connectLayers(lgraph,"conv_225","conv_238");
lgraph = connectLayers(lgraph,"conv_225","conv_239");
lgraph = connectLayers(lgraph,"conv_225","conv_240");
lgraph = connectLayers(lgraph,"conv_225","conv_241");
lgraph = connectLayers(lgraph,"conv_225","addition_10/in2");
lgraph = connectLayers(lgraph,"conv_258","addition_9/in1");
lgraph = connectLayers(lgraph,"conv_259","addition_9/in2");
lgraph = connectLayers(lgraph,"conv_260","addition_9/in3");
lgraph = connectLayers(lgraph,"conv_261","addition_9/in4");
lgraph = connectLayers(lgraph,"conv_262","addition_9/in5");
lgraph = connectLayers(lgraph,"conv_263","addition_9/in6");
lgraph = connectLayers(lgraph,"conv_264","addition_9/in7");
lgraph = connectLayers(lgraph,"conv_265","addition_9/in12");
lgraph = connectLayers(lgraph,"conv_266","addition_9/in10");
lgraph = connectLayers(lgraph,"conv_267","addition_9/in9");
lgraph = connectLayers(lgraph,"conv_268","addition_9/in8");
lgraph = connectLayers(lgraph,"conv_269","addition_9/in11");
lgraph = connectLayers(lgraph,"conv_270","addition_9/in14");
lgraph = connectLayers(lgraph,"conv_271","addition_9/in15");
lgraph = connectLayers(lgraph,"conv_272","addition_9/in16");
lgraph = connectLayers(lgraph,"conv_273","addition_9/in13");
lgraph = connectLayers(lgraph,"layernorm_6","addition_10/in1");
lgraph = connectLayers(lgraph,"relu_6","conv_274");
lgraph = connectLayers(lgraph,"relu_6","conv_279");
lgraph = connectLayers(lgraph,"relu_6","conv_282");
lgraph = connectLayers(lgraph,"relu_6","conv_285");
lgraph = connectLayers(lgraph,"relu_6","conv_288");
lgraph = connectLayers(lgraph,"relu_6","conv_291");
lgraph = connectLayers(lgraph,"relu_6","conv_294");
lgraph = connectLayers(lgraph,"relu_6","conv_297");
lgraph = connectLayers(lgraph,"relu_6","multiplication_28/in2");
lgraph = connectLayers(lgraph,"conv_274","resize3d-output-size_72");
lgraph = connectLayers(lgraph,"conv_274","resize3d-output-size_73");
lgraph = connectLayers(lgraph,"conv_278","multiplication_27/in1");
lgraph = connectLayers(lgraph,"conv_277","multiplication_27/in2");
lgraph = connectLayers(lgraph,"conv_279","resize3d-output-size_74");
lgraph = connectLayers(lgraph,"conv_279","resize3d-output-size_75");
lgraph = connectLayers(lgraph,"conv_281","multiplication_29/in1");
lgraph = connectLayers(lgraph,"conv_280","multiplication_29/in2");
lgraph = connectLayers(lgraph,"conv_282","resize3d-output-size_76");
lgraph = connectLayers(lgraph,"conv_282","resize3d-output-size_77");
lgraph = connectLayers(lgraph,"conv_284","multiplication_30/in1");
lgraph = connectLayers(lgraph,"conv_283","multiplication_30/in2");
lgraph = connectLayers(lgraph,"conv_285","resize3d-output-size_78");
lgraph = connectLayers(lgraph,"conv_285","resize3d-output-size_79");
lgraph = connectLayers(lgraph,"conv_287","multiplication_31/in1");
lgraph = connectLayers(lgraph,"conv_286","multiplication_31/in2");
lgraph = connectLayers(lgraph,"resize3d-output-size_80","addition_11/in1");
lgraph = connectLayers(lgraph,"resize3d-output-size_81","addition_11/in2");
lgraph = connectLayers(lgraph,"resize3d-output-size_82","addition_11/in3");
lgraph = connectLayers(lgraph,"resize3d-output-size_83","addition_11/in4");
lgraph = connectLayers(lgraph,"conv_288","resize3d-output-size_84");
lgraph = connectLayers(lgraph,"conv_288","resize3d-output-size_85");
lgraph = connectLayers(lgraph,"conv_290","multiplication_32/in1");
lgraph = connectLayers(lgraph,"conv_289","multiplication_32/in2");
lgraph = connectLayers(lgraph,"resize3d-output-size_86","addition_11/in5");
lgraph = connectLayers(lgraph,"conv_291","resize3d-output-size_87");
lgraph = connectLayers(lgraph,"conv_291","resize3d-output-size_88");
lgraph = connectLayers(lgraph,"conv_293","multiplication_33/in1");
lgraph = connectLayers(lgraph,"conv_292","multiplication_33/in2");
lgraph = connectLayers(lgraph,"resize3d-output-size_89","addition_11/in6");
lgraph = connectLayers(lgraph,"conv_294","resize3d-output-size_90");
lgraph = connectLayers(lgraph,"conv_294","resize3d-output-size_91");
lgraph = connectLayers(lgraph,"conv_296","multiplication_34/in1");
lgraph = connectLayers(lgraph,"conv_295","multiplication_34/in2");
lgraph = connectLayers(lgraph,"resize3d-output-size_92","addition_11/in7");
lgraph = connectLayers(lgraph,"conv_297","resize3d-output-size_93");
lgraph = connectLayers(lgraph,"conv_297","resize3d-output-size_94");
lgraph = connectLayers(lgraph,"conv_299","multiplication_35/in1");
lgraph = connectLayers(lgraph,"conv_298","multiplication_35/in2");
lgraph = connectLayers(lgraph,"resize3d-output-size_95","addition_11/in8");
lgraph = connectLayers(lgraph,"conv_276","multiplication_28/in1");

% Display the network structure we've constructed above.
plot(lgraph);

%% Train Network
% Train the network using the specified options and training data.
[net, traininfo] = trainNetwork(augimdsTrain,lgraph,opts);
