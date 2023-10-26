%% R^2-TM Generator
% Former Author: JoeyBG;
% Improved Author: JoeyBG;
% Affiliation: Beijing Institute of Technology, Radar Research Lab;
% Date: 2023-4-23;
% Language & Platform: MATLAB R2023a.
% 
% Introduction: 
% This is the very first radar image processing schemes we proposed.
% Stretching the vertical coordinates of a radar range-time image (RTM) 
% from linear to square coordinates is a common image processing method used 
% to improve the visualization of radar images in the range dimension.
%
% Theory in Simple:
% Here's a detailed explanation of the image processing approach to stretch 
% the vertical axis of a radar Range-Time Image (RTM) from linear coordinates 
% to square coordinates:
% 1. Data Acquisition and Preprocessing: 
% First, the RTM is acquired by collecting data using a radar system. 
% The RTM is a two-dimensional image where the horizontal axis represents 
% time and the vertical axis represents distance. Each pixel's intensity 
% corresponds to the echo signal strength at the corresponding position. 
% Before performing the vertical axis stretch, it is common to preprocess 
% the RTM by removing noise, applying background subtraction, and adjusting 
% the dynamic range to enhance image quality and target detection performance.
% 2. Square Coordinate Transformation: 
% To stretch the vertical axis of the RTM from linear coordinates to square 
% coordinates, we can apply a square operation to the vertical distance values. 
% Specifically, for each pixel's vertical coordinate distance value, we square it. 
% This operation increases the visual contrast for targets located at longer 
% distances, making them more prominent in the image.
% 3. Dynamic Range Adjustment: 
% After the vertical axis stretch, it is often beneficial to apply dynamic 
% range adjustment techniques to further enhance the details and target information 
% in the image. This can involve linear or non-linear grayscale stretching, 
% histogram equalization, or other methods to improve contrast and visualization.
% 4. Visualization and Analysis: 
% Once the vertical axis has been stretched to square coordinates, the processed 
% image can be visualized and analyzed. This may involve techniques such as 
% target detection, target tracking, feature extraction, and data analysis 
% to gain further insights and utilize the information present in the image.
%

%% Initialization of Matlab Script
clear all;
close all;
clc;
load("R2TM_D2TM_Clist.mat");
R2TM_Colormap = R2TM_D2TM_Clist;

%% Load the RTM Image into the Workspace
Image_Path = ' '; % Replace the path to your own path that store the RTM.
if Image_Path == ' '
    error('Image_Path is empty. Please replace it with your feature map folder path. Program terminated!');
end
Raw_RTM_Unit = imread(Image_Path);
Raw_RTM = double(rgb2gray(Raw_RTM_Unit)); % Convert the input image into 1-channel double form.
[m,n] = size(Raw_RTM);
Raw_RTM_Normalized = zeros(m,n);
Max_RawRTM = max(max(Raw_RTM));
Min_RawRTM = min(min(Raw_RTM));
for i = 1:m
    for j = 1:n
        Raw_RTM_Normalized(i,j) = (Raw_RTM(i,j)-Min_RawRTM)/(Max_RawRTM-Min_RawRTM);
    end
end % Normalize the input image.

%% Convert R-TM to R^2-TM
Target_m = 70; % The size of resized fast-time axis.
Raw_RTM_Normalized = flip(Raw_RTM_Normalized);
RTM = imresize(Raw_RTM_Normalized,[Target_m,n]);
RTM_Normalized = zeros(Target_m,n);
Max_RTM = max(max(RTM));
Min_RTM = min(min(RTM));
for i = 1:Target_m
    for j = 1:n
        RTM_Normalized(i,j) = (RTM(i,j)-Min_RTM)/(Max_RTM-Min_RTM);
    end
end % Normalize the input image.
R2TM = zeros(Target_m^2,n); % Create the R^2-TM Matrix.
k = 1; % Calculator, which is used to calculate the current number of cells used for ^2-mapping filling.
for i = 1:Target_m
    for j = 1:n
        for k = 1:(2*i-1)
            R2TM((i-1)^2+k,j) = RTM_Normalized(i,j);
        end
    end
end
[m2,n2] = size(R2TM);

%% Plot the Results
% Plot the original RTM in a more colorful way.
figure;
subplot(1,2,1);
imagesc(1-Raw_RTM_Normalized); 
axis tight;
colormap(R2TM_Colormap);
% colorbar;
% xlabel('Time (s)');
% ylabel('Range (m)');
set(gca,'XTick',0:n/4:n);
set(gca,'XTicklabel',{' ',' ',' ',' ',' '});
set(gca,'YTick',0:m/4:m);
set(gca,'YTicklabel',{' ',' ',' ',' ',' '});
set(gca,'FontName','TsangerYuMo W03');
set(gca,'FontSize',20);
set(gca,'ydir','normal');
% title('Processed RTM','FontName','TsangerYuMo W03','FontWeight','bold','FontSize',16);

% Plot the processed R2TM in a more colorful way.
subplot(1,2,2);
imagesc(1-R2TM);
axis tight;
colormap(R2TM_Colormap);
% colorbar;
% xlabel('Time (s)');
% ylabel('Range * Range (m^2)');
set(gca,'XTick',0:n2/4:n2);
set(gca,'XTicklabel',{' ',' ',' ',' ',' '});
set(gca,'YTick',[0,m2/16,m2/4,9*m2/16,m2]);
set(gca,'YTicklabel',{' ',' ',' ',' ',' '});
set(gca,'FontName','TsangerYuMo W03');
set(gca,'FontSize',20);
set(gca,'ydir','normal');
% title('Processed R^2TM','FontName','TsangerYuMo W03','FontWeight','bold','FontSize',16);
