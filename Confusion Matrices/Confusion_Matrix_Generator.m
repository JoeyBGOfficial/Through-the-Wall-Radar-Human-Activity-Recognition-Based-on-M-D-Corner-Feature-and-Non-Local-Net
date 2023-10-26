%% Confusion Matrix Generator
% Former Author: JoeyBG;
% Improved Author: JoeyBG;
% Affiliation: Beijing Institute of Technology, Radar Research Lab;
% Date: 2023-8-23;
% Language & Platform: MATLAB R2023a.
% 
% Introduction: 
% To plot a confusion matrix using MATLAB, you can follow these steps:
% Calculate the confusion matrix: First, you need to obtain the confusion 
% matrix for your classification results. This matrix represents the 
% performance of your classifier by comparing the predicted labels with 
% the true labels. There are several ways to compute the confusion matrix 
% depending on your specific classification problem and the format of your 
% predictions and ground truth labels.
% Normalize the confusion matrix (optional): Optionally, you can normalize 
% the confusion matrix to better visualize the relative proportions of the 
% different classes. Normalization can be done by dividing each element in 
% the confusion matrix by the sum of the row it belongs to.
% Plot the confusion matrix: Once you have the confusion matrix, you can 
% use the MATLAB imagesc function to visualize it as a color-coded image. 
% This function displays matrix data where each element's color represents 
% its value. You can customize the colormap and add colorbar to enhance the 
% visualization.
% In this code, a method for loading the confusion matrix on the GPU is 
% provided, along with a faster approach for plotting the confusion matrix.
%

%% Matlab Initialization
clear all;
close all;
clc;

%% Main Body for Plotting the Confusion Matrix
% Readin the Datas in xlsx files
Name = " "; % Change it to your own confusion matrix's name, the confusion matrix should be a 3 × 3 double array.
if Name == " "
    error("The input of your confusion matrix is empty!");
end
load(strcat(Name,'.mat'));
Datas = int16(Name);
    
% Generate the Label Vector.
Labels = zeros([360,1]);
for i = 1:120
    Labels(i) = 1;
end
for j = 2:3
    for i = (120*j-119):120*j
        Labels(i) = j;
    end
end

% Generate Prediction Vector.
Predicts = zeros([360,1]);
Cal = 1;
for j = 1:3
    for i = 1:3
        Num = Datas(i,j);        
        for k = 1:Num
            Predicts(Cal) = i;
            Cal = Cal+1;
        end
    end
end

% Convert the Datas into GPU arrays.
GPU_Labels = gpuArray(Labels);
GPU_Predicts = gpuArray(Predicts);
Labels_Conf = categorical(Labels);
Predicts_Conf = categorical(Predicts);

% Plot Confusion Matrix.
plotconfusion(Labels_Conf,Predicts_Conf,Name);
set(gca,'Fontname','TsangerYuMo W03','Fontsize',15);

% Save the Figure Above.
set(gcf,'Position', [200 200 750 750]);
exportgraphics(gcf,strcat(Name,' Confusion Matrix.png'),'Resolution',600);
close all;

% % Plot Confusion Matrix in a Faster Way.
% cm = confusionchart(Labels,Predicts);
% cm.Title = Name;
% cm.RowSummary = "row-normalized";
% cm.ColumnSummary = "column-normalized";
% cm.DiagonalColor = "#77AC30";
% cm.OffDiagonalColor = "#A2142F";
% cm.GridVisible = "on";

% % Save the Fast Figure Above.
% set(gcf,'Position', [100 100 1200 400]);
% exportgraphics(gcf,'TWR-ResNeXt Confusion Matrix.png','Resolution',1200);
% close all;
