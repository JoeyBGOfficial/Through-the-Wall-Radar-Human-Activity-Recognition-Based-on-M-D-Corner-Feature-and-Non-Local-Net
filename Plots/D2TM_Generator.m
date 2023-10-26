%% D^2-TM Generator
% Former Author: JoeyBG;
% Improved Author: JoeyBG;
% Affiliation: Beijing Institute of Technology, Radar Research Lab;
% Date: 2023-4-23;
% Language & Platform: MATLAB R2023a.
% 
% Introduction: 
% This is the very first radar image processing schemes we proposed.
% Stretching the vertical coordinates of a radar Doppler-time image (DTM) 
% from linear to square coordinates is a common image processing method used 
% to improve the visualization of radar images in the Doppler dimension.
%
% Theory in Simple:
% Here's a detailed explanation of the image processing approach to stretch 
% the vertical axis of a radar Doppler-Time Image (DTM) from linear coordinates 
% to square coordinates:
% 1. Data Acquisition and Preprocessing: 
% First, the DTM is acquired by collecting data using a radar system. 
% The DTM is a two-dimensional image where the horizontal axis represents 
% time and the vertical axis represents frequency. Each pixel's intensity 
% corresponds to the echo signal strength with the corresponding velocity. 
% Before performing the vertical axis stretch, it is common to preprocess 
% the DTM by removing noise, applying background subtraction, STFT, and adjusting 
% the dynamic range to enhance image quality and target detection performance.
% 2. Square Coordinate Transformation: 
% To stretch the vertical axis of the DTM from linear coordinates to square 
% coordinates, we can apply a square operation to the vertical frequency values.
% The difference between stretching DTM and RTM is that the DTM should be
% cut in two half matrices along 0-Doppler axis. The stretching process
% should be performed on each half matrices. After stretching,
% concatenation process is needed for reconstruction the complete D2TM.
% Specifically, for each pixel's vertical coordinate frequency value, we square it. 
% This operation increases the visual contrast for targets moving in higher
% speeed, making them more prominent in the image.
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
D2TM_Colormap = R2TM_D2TM_Clist;

%% Load the DTM Image into the Workspace
Image_Path = ' '; % Replace the path to your own path that store the DTM.
if Image_Path == ' '
    error('Image_Path is empty. Please replace it with your feature map folder path. Program terminated!');
end
Raw_DTM_Unit = imread(Image_Path);
Raw_DTM = double(rgb2gray(Raw_DTM_Unit)); % Convert the input image into 1-channel double form.
[m,n] = size(Raw_DTM);
Raw_DTM_Normalized = zeros(m,n);
Max_RawDTM = max(max(Raw_DTM));
Min_RawDTM = min(min(Raw_DTM));
for i = 1:m
    for j = 1:n
        Raw_DTM_Normalized(i,j) = (Raw_DTM(i,j)-Min_RawDTM)/(Max_RawDTM-Min_RawDTM);
    end
end % Normalize the input image.

%% Convert D-TM to D^2-TM
Target_m = 70; % The size of resized Doppler axis, which is Suggested be a even number.
Raw_DTM_Normalized = flip(Raw_DTM_Normalized);
DTM = imresize(Raw_DTM_Normalized,[Target_m,n]);
DTM_Normalized = zeros(Target_m,n);
Max_DTM = max(max(DTM));
Min_DTM = min(min(DTM));
for i = 1:Target_m
    for j = 1:n
        DTM_Normalized(i,j) = (DTM(i,j)-Min_DTM)/(Max_DTM-Min_DTM);
    end
end % Normalize the input image.
D2TM = zeros(Target_m^2/2,n); % Create the D^2-TM Matrix.
D2TM_Top = zeros(Target_m^2/4,n);
D2TM_Bottom = zeros(Target_m^2/4,n);
k = 1; % Calculator, which is used to calculate the current number of cells used for ^2-mapping filling.
for i = 1:Target_m/2
    for j = 1:n
        for k = 1:(2*i-1)
            D2TM_Bottom((i-1)^2+k,j) = DTM_Normalized(Target_m/2+1-i,j);
        end
    end
end 
for i = 1:Target_m/2
    for j = 1:n
        for k = 1:(2*i-1)
            D2TM_Top((i-1)^2+k,j) = DTM_Normalized(Target_m/2+i,j);
        end
    end
end 
[m_Top,n_Top] = size(D2TM_Top);
[m_Bottom,n_Bottom] = size(D2TM_Bottom);
for i = 1:m_Bottom
    D2TM(i,:) = D2TM_Bottom(m_Bottom+1-i,:);
end
for i = 1:m_Top
    D2TM(m_Bottom+i,:) = D2TM_Top(i,:);
end
[m2,n2] = size(D2TM);

%% Plot the Results
% Plot the original DTM in a more colorful way.
figure;
subplot(1,2,1);
imagesc(1-Raw_DTM_Normalized);
axis tight;
colormap(D2TM_Colormap);
% colorbar;
% xlabel('Time (s)');
% ylabel('Doppler (Hz)');
set(gca,'XTick',0:n/4:n);
set(gca,'XTicklabel',{' ',' ',' ',' ',' '});
set(gca,'YTick',0:m/4:m);
set(gca,'YTicklabel',{' ',' ',' ',' ',' '});
set(gca,'FontName','TsangerYuMo W03');
set(gca,'FontSize',20);
set(gca,'ydir','normal');
% title('Processed DTM','FontName','TsangerYuMo W03','FontWeight','bold','FontSize',16);

% Plot the processed D2TM in a more colorful way.
subplot(1,2,2);
imagesc(1-D2TM);
axis tight;
colormap(D2TM_Colormap);
% colorbar;
% xlabel('Time (s)');
% ylabel('Doppler * Doppler (Free Space: (Hz*51.33)^2, TWR: Hz^2)');
set(gca,'XTick',0:n2/4:n2);
set(gca,'XTicklabel',{' ',' ',' ',' ',' '});
set(gca,'YTick',[0,3*m2/8,m2/2,5*m2/8,m2]);
set(gca,'YTicklabel',{' ',' ',' ',' ',' '});
set(gca,'FontName','TsangerYuMo W03');
set(gca,'FontSize',20);
set(gca,'ydir','normal');
% title('Processed D^2TM','FontName','TsangerYuMo W03','FontWeight','bold','FontSize',16);
