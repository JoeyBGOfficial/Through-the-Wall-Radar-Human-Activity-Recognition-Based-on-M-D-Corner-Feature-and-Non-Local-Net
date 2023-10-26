%% A Program for Visualizing T-SNE Dimension Reduction Results
% Former Author: JoeyBG;
% Improved By: JoeyBG;
% Affiliation: Beijing Institute of Technology, Radar Research Lab;
% Date: 2023-8-14;
% Language & Platform: MATLAB R2023a.
%
% Introduction: Here we provide a series of data reading codes, including
% data formats from images to Mat signal files, most of which come from
% open source software or databases. Please refer to the original author
% when using. In addition, we select some from each category of the dataset
% and implemented the display of downscaled and 2D scatterplots using the
% T-SNE approach.
% 
% Theory in Simple:
% Here's an explanation of the T-SNE approach along with the mathematical formulas involved:
% 1. Define similarity between data points: 
% T-SNE starts by defining a similarity measure between pairs of high-dimensional data points. 
% This similarity is typically computed using a Gaussian kernel or a Student's t-distribution. 
% The similarity between two data points xi and xj is denoted as p(i|j) or p(j|i).
% 2. Compute pairwise similarities: 
% T-SNE computes the pairwise similarities between all data points in the high-dimensional space. 
% The similarity measure ensures that similar points have higher probabilities of being selected.
% 3. Define similarity in the lower-dimensional space: 
% T-SNE then defines a similarity measure in the lower-dimensional space 
% for the mapped points yi and yj. This similarity is denoted as q(i|j) or q(j|i).
% 4. Minimize the divergence: 
% The goal of T-SNE is to minimize the divergence between the pairwise similarities 
% in the high-dimensional space (p(i|j)) and the pairwise similarities in the 
% lower-dimensional space (q(i|j)). 
% This is achieved by minimizing the Kullback-Leibler (KL) divergence between the two distributions.
% 5. Define the cost function: 
% The cost function in T-SNE combines the KL divergence term to be minimized 
% and a regularization term to prevent overcrowding of points in the low-dimensional space. 
% The cost function is defined as:
% C = KL(P || Q) = Σi Σj p(i|j) log(p(i|j) / q(i|j))
% Here, P represents the pairwise similarities in the high-dimensional space, 
% and Q represents the pairwise similarities in the lower-dimensional space.
% 6. Gradient descent optimization: 
% T-SNE uses gradient descent optimization to minimize the cost function iteratively. 
% It computes the gradient of the cost function with respect to the embedded 
% points yi and adjusts their positions in the low-dimensional space to reduce the cost.
% 7. Iteratively update and visualize: 
% The optimization process is repeated for multiple iterations, and at each iteration, 
% the embedded points are updated based on the gradients computed. 
% Finally, the lower-dimensional points can be visualized using a scatter 
% plot or other visualization techniques.
%
% Citation:
% [1] van der Maaten, Laurens, and Geoffrey Hinton. "Visualizing Data using
% t-SNE." J. Machine Learning Research 9, 2008, pp. 2579–2605.
% [2] van der Maaten, Laurens. Barnes-Hut-SNE. arXiv:1301.3342 [cs.LG],
% 2013.
% [3] Jacobs, Robert A. "Increased rates of convergence through learning
% rate adaptation." Neural Networks 1.4, 1988, pp. 295–307.
% [4] Wattenberg, Martin, Fernanda Viégas, and Ian Johnson. "How to Use
% t-SNE Effectively." Distill, 2016. Available at How to Use t-SNE
% Effectively.

%% Preparations for Matlab Script
clc;
clear all;
close all;
load("TSNE_CList.mat");
TSNE_Colormap = TSNE_Clist;

%% Set the Categories and Datapath for Program Running
Image_Folder = ' ';  % Replace with your image folder path.
if Image_Folder == ' '
    error('Image_Folder is empty. Please replace it with your feature map folder path. Program terminated!');
end
categories = {'S1', 'S2', 'S3'};  % Replace with your category tags.
imageNumber = 60; % The number of images used for generating scatter plot.
img_resize = 256; % Resize the input images to [img_resize img_resize].

% Readin the images.
images = [];
labels = [];
for i = 1:length(categories)
    folderPath = fullfile(Image_Folder, categories{i});
    imageFiles = dir(fullfile(folderPath, '*.png'));  % Assuming the image file is in PNG format, if there are other formats, please modify accordingly.
    numImages = length(imageFiles);

    % Read the images one by one and adding labels.
    for j = 21:imageNumber+20
        imagePath = fullfile(folderPath, imageFiles(j).name);
        img = imread(imagePath);
        img = imresize(img,[img_resize img_resize]);

        % % 1D feature reduction.
        % img  = double(rgb2gray(img));
        % coeff = pca(img);
        % numDimensions = 1;  % Set the number of dimensions after dimensionality reduction.
        % reducedFeatures = img * coeff(:, 1:numDimensions);
        % img = reducedFeatures;

        images = [images; img(:)'];
        labels = [labels; i];
    end
end

%% Use T-SNE for Feature Dimension Reduction Analysis
% rng(0);  % Set the random seed to ensure the reproducing of the work.
imagesDouble = double(images); % Convert the images variable to float.
mappedX = tsne(imagesDouble, 'Algorithm', 'exact', 'NumDimensions', 3); % Main body of T-SNE.

% Generation of the 3-D scattering plot.
figure;
mappedX = (mapminmax(mappedX',0,1))';% Linear normalization.
scatter3(mappedX(:, 1), mappedX(:, 2), mappedX(:, 3), 75, labels, 'filled');
% colormap(jet(length(categories)));
colormap(TSNE_Colormap);
% colorbar;
set(gca,'FontName','Times New Roman');
set(gca,'FontSize',34);
xlabel('X','FontSize',34);
ylabel('Y','FontSize',34);
zlabel('Z','FontSize',34);
title('Proposed Method','FontSize',38,'FontWeight','bold','FontName','Times New Roman');